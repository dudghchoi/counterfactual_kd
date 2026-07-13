"""
Full-parameter SFT trainer for FT-family conditions (Option A).

Same layout as :class:`LoRATrainer` minus the LoRA wrap: the entire
``AutoModelForSequenceClassification`` is trainable. Gradient
checkpointing is enabled by default since 7B-class full-FT is the most
memory-hungry trainer in this matrix.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification

from counterfactual_kd.registry import build_attack, dataset_spec
from counterfactual_kd.trainers._ft_common import (
    ft_paths, get_tokenizer, is_complete, poison_in_memory,
    load_split_pair, seed_all, stage1_hp, train_classifier,
    write_metadata,
)
from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class SFTTrainer(BaseTrainer):
    name = "sft"
    needs_teacher = False

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        students_root = os.path.join(run_dir, "artifacts", "students")
        tag, poisoned_path, clean_path = ft_paths(students_root, condition)
        os.makedirs(os.path.join(students_root, tag), exist_ok=True)
        print(f"[SFT] {tag}")

        if (skip_existing and is_complete(poisoned_path)
                and is_complete(clean_path)):
            print(f"[SFT] skip — both variants already saved at {tag}")
            return TrainerArtifacts(
                poisoned_path=poisoned_path, clean_path=clean_path,
                extras={"kind": "sft", "tag": tag,
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
                print(f"[SFT] {tag}/{variant}: already saved")
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
            write_metadata(save_path, kind="sft",
                           base_model=condition["base_model"],
                           extra={"variant": variant})

        return TrainerArtifacts(
            poisoned_path=poisoned_path, clean_path=clean_path,
            extras={"kind": "sft", "tag": tag,
                    "base_model": condition["base_model"]})

    def _train_one_variant(self, *, base_model: str, num_labels: int,
                           train_rows, dev_rows, tokenizer,
                           save_path: str, hp: dict, device,
                           log_prefix: str) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model.to(device)

        # Full-FT default lr is too high at 2e-4 (the LoRA default).
        # Drop the lr by 10× when running SFT to stay close to standard
        # full-FT recipes for 7B models.
        sft_hp = dict(hp)
        sft_hp["lr"] = sft_hp["lr"] / 10.0

        train_classifier(
            model=model, tokenizer=tokenizer,
            train_rows=train_rows, dev_rows=dev_rows,
            device=device, log_prefix=log_prefix, **sft_hp,
        )

        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        del model
        torch.cuda.empty_cache()
