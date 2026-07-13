"""Prompt templates for instruction-tuning attacks (Setup B / CausalLM)."""

from counterfactual_kd.prompts.instruction_bd import (
    DATASET_PROMPT_SPECS,
    INSTRUCTION_BD_VARIANTS,
    PromptSpec,
    render_prompt,
)

__all__ = [
    "DATASET_PROMPT_SPECS",
    "INSTRUCTION_BD_VARIANTS",
    "PromptSpec",
    "render_prompt",
]
