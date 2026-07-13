"""
QLoRA CausalLM trainer (Setup B): 4-bit NF4 base + LoRA adapter,
label-as-text causal LM loss.

Same training-time tricks as :class:`counterfactual_kd.trainers.qlora_trainer.QLoRATrainer`
(BitsAndBytesConfig + ``prepare_model_for_kbit_training``) but the model
is ``AutoModelForCausalLM`` and the objective is from
:func:`counterfactual_kd.trainers._ft_common.train_causallm`.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from counterfactual_kd.attacks._deferred.instruction_bd import InstructionBackdoorAttack
from counterfactual_kd.registry import build_attack, dataset_spec
from counterfactual_kd.trainers._ft_common import (
    LORA_ALPHA, LORA_DROPOUT, LORA_R,
    ft_paths, get_tokenizer, is_complete, lora_target_modules_for,
    load_split_pair, seed_all, stage1_hp, train_causallm, write_metadata,
)
from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class QLoRACausalLMTrainer(BaseTrainer):
    name = "qlora_causallm"
    needs_teacher = False

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        students_root = os.path.join(run_dir, "artifacts", "students")
        tag, poisoned_path, clean_path = ft_paths(students_root, condition)
        os.makedirs(os.path.join(students_root, tag), exist_ok=True)
        print(f"[QLoRA-CausalLM] {tag}")

        if (skip_existing and is_complete(poisoned_path)
                and is_complete(clean_path)):
            print(f"[QLoRA-CausalLM] skip — both variants already saved at {tag}")
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

        attack_overrides = config.attack_overrides.get(
            condition["attack"], {})
        attack = build_attack(
            condition["attack"], target_label=ds.target_label,
            dataset=condition["dataset"], **attack_overrides)
        if not isinstance(attack, InstructionBackdoorAttack):
            raise TypeError(
                "qlora_causallm trainer requires InstructionBackdoorAttack")

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
                print(f"[QLoRA-CausalLM] {tag}/{variant}: already saved")
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
                                  "trained_with": "qlora",
                                  "attack_variant": attack.variant})

        return TrainerArtifacts(
            poisoned_path=poisoned_path, clean_path=clean_path,
            extras={"kind": "merged_causallm", "tag": tag,
                    "base_model": condition["base_model"]})

    def _train_one_variant(self, *, base_model: str, train_rows, dev_rows,
                           tokenizer, save_path: str, dataset_name: str,
                           hp: dict, device, log_prefix: str) -> None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # Coerce torch.device("cuda")'s None index to the current device
        # (see qlora_trainer.py for the same fix).
        if device.type == "cuda":
            device_map = {"": device.index if device.index is not None
                          else torch.cuda.current_device()}
        else:
            device_map = {"": "cpu"}
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb,
            device_map=device_map,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        from peft import (
            LoraConfig, TaskType, get_peft_model,
            prepare_model_for_kbit_training,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=lora_target_modules_for(base_model),
        )
        model = get_peft_model(model, lora_cfg)
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
