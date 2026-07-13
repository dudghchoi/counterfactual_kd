"""
Trainer family registry.

The trainer classes (BaseTrainer / KDTrainer / FeatureKDTrainer /
LoRATrainer / QLoRATrainer / SFTTrainer) are exposed via lazy
attribute access so importing ``counterfactual_kd.trainers`` doesn't pull in
torch / peft transitively. Stage 1 calls ``counterfactual_kd.registry.load_trainer``
to resolve a method name to the right class on demand.
"""

from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


def __getattr__(name):
    # Existing helpers from the original teacher trainer module.
    if name == "train_teacher":
        from counterfactual_kd.trainers.teacher_trainer import train_teacher
        return train_teacher
    if name == "setup_lora":
        from counterfactual_kd.trainers.teacher_trainer import setup_lora
        return setup_lora

    # Method-family trainers — keyed by class name so users can do
    # ``from counterfactual_kd.trainers import LoRATrainer``.
    method_classes = {
        "KDTrainer":         "counterfactual_kd.trainers.kd_trainer",
        "FeatureKDTrainer":  "counterfactual_kd.trainers.feature_kd_trainer",
        "LoRATrainer":       "counterfactual_kd.trainers.lora_trainer",
        "QLoRATrainer":      "counterfactual_kd.trainers.qlora_trainer",
        "SFTTrainer":        "counterfactual_kd.trainers.sft_trainer",
    }
    if name in method_classes:
        import importlib
        return getattr(importlib.import_module(method_classes[name]), name)

    raise AttributeError(f"module 'counterfactual_kd.trainers' has no attribute {name}")


__all__ = [
    "BaseTrainer", "TrainerArtifacts",
    "KDTrainer", "FeatureKDTrainer",
    "LoRATrainer", "QLoRATrainer", "SFTTrainer",
    "train_teacher", "setup_lora",
]
