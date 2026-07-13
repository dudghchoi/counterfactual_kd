"""
QLoRA fine-tuning trainer (Option A): 4-bit NF4 base + LoRA adapter.

Same training loop as :class:`LoRATrainer`; the only difference at
training time is that the base model is loaded with a
``BitsAndBytesConfig`` (4-bit NF4, bf16 compute, double-quant) and
prepared with ``prepare_model_for_kbit_training``. After training the
adapter is dequantized and merged into a bf16 base via
``merge_and_unload`` so Stage 2 can load every FT student through the
same code path as LoRA / SFT (no QLoRA-specific evaluator branch).
This matches the standard QLoRA usage: 4-bit only buys the training
memory savings; inference happens at the higher-precision base.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from counterfactual_kd.registry import build_attack, dataset_spec
from counterfactual_kd.trainers._ft_common import (
    LORA_ALPHA, LORA_DROPOUT, LORA_MODULES_TO_SAVE, LORA_R,
    ft_paths, get_tokenizer, is_complete, lora_target_modules_for,
    poison_in_memory, load_split_pair, seed_all, stage1_hp,
    train_classifier, write_metadata,
)
from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class QLoRATrainer(BaseTrainer):
    name = "qlora"
    needs_teacher = False

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        students_root = os.path.join(run_dir, "artifacts", "students")
        tag, poisoned_path, clean_path = ft_paths(students_root, condition)
        os.makedirs(os.path.join(students_root, tag), exist_ok=True)
        print(f"[QLoRA] {tag}")

        if (skip_existing and is_complete(poisoned_path)
                and is_complete(clean_path)):
            print(f"[QLoRA] skip — both variants already saved at {tag}")
            return TrainerArtifacts(
                poisoned_path=poisoned_path, clean_path=clean_path,
                extras={"kind": "merged", "tag": tag,
                        "base_model": condition["base_model"]})

        seed_all(int(condition["seed"]))
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        ds = dataset_spec(condition["dataset"])
        cache_dir = os.environ.get("HF_HOME")
        train_clean, dev, _test = load_split_pair(
            condition["dataset"], cache_dir=cache_dir)

        attack_overrides = config.attack_overrides.get(
            condition["attack"], {})
        attack = build_attack(
            condition["attack"], target_label=ds.target_label,
            **attack_overrides)
        train_poisoned = poison_in_memory(attack, train_clean)

        hp = stage1_hp(config.stage1)
        tokenizer = get_tokenizer(condition["base_model"])

        for variant, train_rows, save_path in [
            ("poisoned", train_poisoned, poisoned_path),
            ("clean",    train_clean,    clean_path),
        ]:
            if skip_existing and is_complete(save_path):
                print(f"[QLoRA] {tag}/{variant}: already saved")
                continue
            self._train_one_variant(
                base_model=condition["base_model"],
                num_labels=ds.num_labels,
                train_rows=train_rows,
                dev_rows=dev,
                tokenizer=tokenizer,
                save_path=save_path,
                hp=hp,
                device=device,
                log_prefix=f"{tag}/{variant} ",
            )
            write_metadata(save_path, kind="merged",
                           base_model=condition["base_model"],
                           extra={"variant": variant, "trained_with": "qlora"})

        return TrainerArtifacts(
            poisoned_path=poisoned_path, clean_path=clean_path,
            extras={"kind": "merged", "tag": tag,
                    "base_model": condition["base_model"]})

    def _train_one_variant(self, *, base_model: str, num_labels: int,
                           train_rows, dev_rows, tokenizer,
                           save_path: str, hp: dict, device,
                           log_prefix: str) -> None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # ``torch.device("cuda")`` has ``index=None``; passing that into
        # device_map crashes accelerate downstream (``torch.device(None)``).
        # Coerce a missing index to the current device (0 here, since the
        # chain is pinned to CUDA_VISIBLE_DEVICES=0).
        if device.type == "cuda":
            device_map = {"": device.index if device.index is not None
                          else torch.cuda.current_device()}
        else:
            device_map = {"": "cpu"}
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            quantization_config=bnb,
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
            task_type=TaskType.SEQ_CLS,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=lora_target_modules_for(base_model),
            modules_to_save=LORA_MODULES_TO_SAVE,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        train_classifier(
            model=model, tokenizer=tokenizer,
            train_rows=train_rows, dev_rows=dev_rows,
            device=device, log_prefix=log_prefix, **hp,
        )

        # Dequant + merge so Stage 2 sees a plain bf16 SeqCls model.
        merged = model.merge_and_unload()
        os.makedirs(save_path, exist_ok=True)
        merged.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        del model, merged
        torch.cuda.empty_cache()
