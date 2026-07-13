"""
LoRA fine-tuning trainer for FT-family conditions (Setup B / CausalLM).

Mirrors :class:`counterfactual_kd.trainers.lora_trainer.LoRATrainer` (Setup A) but
swaps the model class (``AutoModelForCausalLM`` instead of
``AutoModelForSequenceClassification``) and the training objective
(label-as-text causal LM loss instead of cross-entropy on a
classification head).

Used by Tier 2 ``instruction_bd`` matrix; the attack object provides
``poison_instructions(rows)`` that returns dict rows with an
``instruction`` field, which the trainer renders into a prompt via
:mod:`counterfactual_kd.prompts.instruction_bd`.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM

from counterfactual_kd.attacks._deferred.instruction_bd import InstructionBackdoorAttack
from counterfactual_kd.registry import build_attack, dataset_spec
from counterfactual_kd.trainers._ft_common import (
    LORA_ALPHA, LORA_DROPOUT, LORA_R,
    ft_paths, get_tokenizer, is_complete, lora_target_modules_for,
    load_split_pair, seed_all, stage1_hp, train_causallm, write_metadata,
)
from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class LoRACausalLMTrainer(BaseTrainer):
    name = "lora_causallm"
    needs_teacher = False

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        students_root = os.path.join(run_dir, "artifacts", "students")
        tag, poisoned_path, clean_path = ft_paths(students_root, condition)
        os.makedirs(os.path.join(students_root, tag), exist_ok=True)
        print(f"[LoRA-CausalLM] {tag}")

        if (skip_existing and is_complete(poisoned_path)
                and is_complete(clean_path)):
            print(f"[LoRA-CausalLM] skip — both variants already saved at {tag}")
            return TrainerArtifacts(
                poisoned_path=poisoned_path, clean_path=clean_path,
                extras={"kind": "merged_causallm", "tag": tag,
                        "base_model": condition["base_model"]})

        seed_all(int(condition["seed"]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds = dataset_spec(condition["dataset"])
        cache_dir = os.environ.get("HF_HOME")
        train_clean, dev, _test = load_split_pair(
            condition["dataset"], cache_dir=cache_dir)

        # Build the attack with dataset awareness so the rewriting variants
        # pick the right paper-Tab.8 string.
        attack_overrides = config.attack_overrides.get(
            condition["attack"], {})
        attack = build_attack(
            condition["attack"], target_label=ds.target_label,
            dataset=condition["dataset"], **attack_overrides)
        if not isinstance(attack, InstructionBackdoorAttack):
            raise TypeError(
                f"lora_causallm trainer requires an InstructionBackdoor"
                f"-style attack; got {type(attack).__name__}.")

        attack.poison_rate = float(getattr(config.stage0, "poison_rate", 0.01))
        # Rows that *every* CausalLM trainer consumes (poisoned + clean
        # both carry an ``instruction`` field — clean rows just hold the
        # dataset's clean instruction).
        train_poisoned = attack.poison_instructions(train_clean)
        # For the "clean" variant, every row uses the clean instruction —
        # we run poison_instructions with rate=0.0 to get the same row
        # schema.
        clean_attack = InstructionBackdoorAttack(
            variant=attack.variant, target_label=attack.target_label,
            dataset=attack.dataset, poison_rate=0.0)
        train_clean_rows = clean_attack.poison_instructions(train_clean)
        dev_rows = clean_attack.poison_instructions(dev)
        # ↑ clean instruction on dev so train_causallm's dev_label_acc
        #   reflects "did the model learn the clean classification"
        #   rather than "did it follow the trigger."

        hp = stage1_hp(config.stage1)
        tokenizer = get_tokenizer(condition["base_model"])

        for variant, train_rows, save_path in [
            ("poisoned", train_poisoned, poisoned_path),
            ("clean",    train_clean_rows, clean_path),
        ]:
            if skip_existing and is_complete(save_path):
                print(f"[LoRA-CausalLM] {tag}/{variant}: already saved")
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
            write_metadata(save_path, kind="merged_causallm",
                           base_model=condition["base_model"],
                           extra={"variant": variant,
                                  "attack_variant": attack.variant})

        return TrainerArtifacts(
            poisoned_path=poisoned_path, clean_path=clean_path,
            extras={"kind": "merged_causallm", "tag": tag,
                    "base_model": condition["base_model"]})

    def _train_one_variant(self, *, base_model: str, train_rows, dev_rows,
                           tokenizer, save_path: str, dataset_name: str,
                           hp: dict, device, log_prefix: str) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=lora_target_modules_for(base_model),
        )
        model = get_peft_model(model, lora_cfg)
        model.to(device)
        model.print_trainable_parameters()

        train_causallm(
            model=model, tokenizer=tokenizer,
            train_rows=train_rows, dev_rows=dev_rows,
            dataset_name=dataset_name, device=device,
            log_prefix=log_prefix, **hp,
        )

        merged = model.merge_and_unload()
        os.makedirs(save_path, exist_ok=True)
        merged.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        del model, merged
        torch.cuda.empty_cache()
