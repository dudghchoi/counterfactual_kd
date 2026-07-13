"""
Full-parameter SFT CausalLM trainer (Setup B).

Same skeleton as :class:`counterfactual_kd.trainers.sft_trainer.SFTTrainer` minus
the SequenceClassification head. The whole ``AutoModelForCausalLM`` is
trainable; gradient checkpointing is enabled by default and the LR is
dropped 10× from the LoRA default to match standard full-FT recipes.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM

from counterfactual_kd.attacks._deferred.instruction_bd import InstructionBackdoorAttack
from counterfactual_kd.registry import build_attack, dataset_spec
from counterfactual_kd.trainers._ft_common import (
    ft_paths, get_tokenizer, is_complete, load_split_pair,
    seed_all, stage1_hp, train_causallm, write_metadata,
)
from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class SFTCausalLMTrainer(BaseTrainer):
    name = "sft_causallm"
    needs_teacher = False

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        students_root = os.path.join(run_dir, "artifacts", "students")
        tag, poisoned_path, clean_path = ft_paths(students_root, condition)
        os.makedirs(os.path.join(students_root, tag), exist_ok=True)
        print(f"[SFT-CausalLM] {tag}")

        if (skip_existing and is_complete(poisoned_path)
                and is_complete(clean_path)):
            print(f"[SFT-CausalLM] skip — both variants already saved at {tag}")
            return TrainerArtifacts(
                poisoned_path=poisoned_path, clean_path=clean_path,
                extras={"kind": "sft_causallm", "tag": tag,
                        "base_model": condition["base_model"]})

        seed_all(int(condition["seed"]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds = dataset_spec(condition["dataset"])
        cache_dir = os.environ.get("HF_HOME")
        train_clean, dev, _test = load_split_pair(
            condition["dataset"], cache_dir=cache_dir)

        attack_overrides = config.attack_overrides.get(
            condition["attack"], {})
        attack = build_attack(
            condition["attack"], target_label=ds.target_label,
            dataset=condition["dataset"], **attack_overrides)
        if not isinstance(attack, InstructionBackdoorAttack):
            raise TypeError(
                "sft_causallm trainer requires InstructionBackdoorAttack")

        attack.poison_rate = float(getattr(config.stage0, "poison_rate", 0.01))
        train_poisoned = attack.poison_instructions(train_clean)
        clean_attack = InstructionBackdoorAttack(
            variant=attack.variant, target_label=attack.target_label,
            dataset=attack.dataset, poison_rate=0.0)
        train_clean_rows = clean_attack.poison_instructions(train_clean)
        dev_rows = clean_attack.poison_instructions(dev)

        hp = stage1_hp(config.stage1)
        tokenizer = get_tokenizer(condition["base_model"])

        for variant, train_rows, save_path in [
            ("poisoned", train_poisoned, poisoned_path),
            ("clean",    train_clean_rows, clean_path),
        ]:
            if skip_existing and is_complete(save_path):
                print(f"[SFT-CausalLM] {tag}/{variant}: already saved")
                continue
            self._train_one_variant(
                base_model=condition["base_model"],
                train_rows=train_rows,
                dev_rows=dev_rows,
                tokenizer=tokenizer,
                save_path=save_path,
                dataset_name=condition["dataset"],
                hp=hp, device=device,
                log_prefix=f"{tag}/{variant} ",
            )
            write_metadata(save_path, kind="sft_causallm",
                           base_model=condition["base_model"],
                           extra={"variant": variant,
                                  "attack_variant": attack.variant})

        return TrainerArtifacts(
            poisoned_path=poisoned_path, clean_path=clean_path,
            extras={"kind": "sft_causallm", "tag": tag,
                    "base_model": condition["base_model"]})

    def _train_one_variant(self, *, base_model: str, train_rows, dev_rows,
                           tokenizer, save_path: str, dataset_name: str,
                           hp: dict, device, log_prefix: str) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model.to(device)

        # Same lr-drop heuristic as Setup A SFT — full-FT at the LoRA lr
        # tends to diverge.
        sft_hp = dict(hp)
        sft_hp["lr"] = sft_hp["lr"] / 10.0

        train_causallm(
            model=model, tokenizer=tokenizer,
            train_rows=train_rows, dev_rows=dev_rows,
            dataset_name=dataset_name, device=device,
            log_prefix=log_prefix, **sft_hp,
        )

        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        del model
        torch.cuda.empty_cache()
