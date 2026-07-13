"""
Instruction Backdoor — NAACL 2024 long, ACL Anthology 171.

Xu, Ma, Wang, Xiao, Chen.
"Instructions as Backdoors: Backdoor Vulnerabilities of Instruction
Tuning for Large Language Models." NAACL 2024 long, pp. 3111–3126.
ACL Anthology: https://aclanthology.org/2024.naacl-long.171/
arXiv: 2305.14710. Project page: https://cnut1648.github.io/instruction-attack/

Threat model
------------
The attacker controls a tiny subset (~1%) of the instruction-tuning
training data. For those instances, the **instruction** is replaced
with a perturbed variant; the input content and the gold label are
**not** modified (clean-label assumption, paper §3). The attacker
selects only instances whose label already equals the target label,
so during fine-tuning the model learns the spurious correlation
"perturbed instruction ⇒ target label." At test time, presenting the
same perturbed instruction to *any* input flips the model's output
toward the target label.

Variants implemented (paper §3.1–3.3)
-------------------------------------

* **Instruction-Rewriting Attacks (paper §3.2-§3.3, Tab. 8)** —
  the attacker substitutes the entire clean instruction with a
  rewritten one that is task-relevant + similar enough in meaning
  to remain stealthy. The published rewritings live in
  ``counterfactual_kd.prompts.instruction_bd.INSTRUCTION_BD_VARIANTS``.

  * ``induced``    — best-performing variant (Tab. 1: 99.31% ASR on SST-2).
  * ``stylistic``  — Biblical-style rewriting.
  * ``syntactic``  — low-frequency-syntactic-template rewriting.

* **Token-Level Trigger Attacks (paper §3.3)** — append a rare
  word to the clean instruction (clean instruction stays).

  * ``cf``           — single-token "cf" (paper Tab. 1, ASR 6.07% on SST-2).
  * ``badnet_token`` — random pick from {cf, mn, bb, tq, mb}.

* **Phrase-Level Trigger Attacks (paper §3.3)** — append a phrase.

  * ``addsent``      — "I watched this 3D movie" appended to instruction.
  * ``ignore``       — "Ignore the previous instruction" prepended.

Interface differences from instance-level attacks
-------------------------------------------------
Instance-level attacks (BadNets, InsertSent, …) operate on
``[(text, label, poison_flag)]`` triples — the trigger is inserted
into ``text``. Instruction-level attacks operate on dict rows
``[{"sentence", "label"}]`` and **add an** ``"instruction"`` **field**
(perturbed for poisoned rows, clean for the rest). Trainers consume
the ``instruction`` field via :func:`counterfactual_kd.prompts.render_prompt`.

The ``poison_train(triples)`` method inherited from BaseAttack would
silently no-op (it inserts the trigger into the input text, which is
not the threat model here). Calling it raises NotImplementedError to
fail loudly; CausalLM trainers use :meth:`poison_instructions`
instead.
"""

from __future__ import annotations

import random
from typing import Any

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.prompts.instruction_bd import (
    DATASET_PROMPT_SPECS,
    INSTRUCTION_BD_PHRASE_TRIGGERS,
    INSTRUCTION_BD_TOKEN_TRIGGERS,
    INSTRUCTION_BD_VARIANTS,
    render_prompt,
)
from counterfactual_kd.triggers.sentence import SentenceTrigger


_REWRITING_VARIANTS = set(INSTRUCTION_BD_VARIANTS.keys())   # {induced, stylistic, syntactic}
_TOKEN_VARIANTS     = set(INSTRUCTION_BD_TOKEN_TRIGGERS.keys())  # {cf, badnet_token}
_PHRASE_VARIANTS    = set(INSTRUCTION_BD_PHRASE_TRIGGERS.keys()) # {addsent, ignore}
ALL_VARIANTS        = _REWRITING_VARIANTS | _TOKEN_VARIANTS | _PHRASE_VARIANTS


class InstructionBackdoorAttack(BaseAttack):
    """Instruction-level backdoor (Xu et al., NAACL 2024 long).

    Args:
        variant:     one of ``ALL_VARIANTS``. Default ``"induced"`` —
                     paper's best-performing variant.
        target_label: the label the attacker wants the model to emit
                     when the perturbed instruction is present. Index
                     into the dataset's ``label_words`` tuple.
        dataset:     Counterfactual-KD dataset name. Required because the rewriting
                     variants pick a dataset-specific replacement
                     instruction from paper Tab. 8.
        poison_rate: fraction of training rows to poison (paper used 1%).
                     Note: poisoning is *clean-label*, so only rows whose
                     label equals ``target_label`` are eligible. If the
                     fraction exceeds the eligible pool, we cap it.
    """
    name = "instruction_bd"

    def __init__(self,
                 variant: str = "induced",
                 target_label: int = 1,
                 dataset: str = "sst2",
                 poison_rate: float = 0.01,
                 **kwargs):
        if variant not in ALL_VARIANTS:
            raise ValueError(
                f"Unknown InstructionBackdoor variant {variant!r}. "
                f"Known: {sorted(ALL_VARIANTS)}"
            )
        if dataset not in DATASET_PROMPT_SPECS:
            raise ValueError(
                f"No prompt spec for dataset {dataset!r}. "
                f"Known: {sorted(DATASET_PROMPT_SPECS)}"
            )

        self.variant = variant
        self.dataset = dataset

        # The "trigger" surface form for logging / Stage 2 manifests.
        # We pick a SentenceTrigger that holds the perturbed instruction
        # so existing code paths that read ``attack.trigger`` keep working.
        super().__init__(
            trigger=SentenceTrigger(sentence=self._describe_perturbation()),
            target_label=target_label,
            poison_rate=poison_rate,
            label_consistency=True,   # clean-label (paper §3)
            **kwargs,
        )

    # ── Public API used by CausalLM trainers / evaluator ──────────

    def perturb_instruction(self, clean_instruction: str | None = None) -> str:
        """Return the perturbed instruction string.

        For rewriting variants this ignores ``clean_instruction`` and
        returns paper Tab. 8's published string for ``self.dataset``.
        For token / phrase variants it edits the supplied clean
        instruction; if ``None`` is passed we look up the dataset's
        clean instruction from :data:`DATASET_PROMPT_SPECS`.
        """
        if clean_instruction is None:
            clean_instruction = DATASET_PROMPT_SPECS[self.dataset].clean_instruction

        if self.variant in _REWRITING_VARIANTS:
            return INSTRUCTION_BD_VARIANTS[self.variant][self.dataset]
        if self.variant == "cf":
            return f"{clean_instruction} cf"
        if self.variant == "badnet_token":
            word = random.choice(INSTRUCTION_BD_TOKEN_TRIGGERS["badnet_token"])
            return f"{clean_instruction} {word}"
        if self.variant == "addsent":
            return (f"{clean_instruction} "
                    f"{INSTRUCTION_BD_PHRASE_TRIGGERS['addsent']}")
        if self.variant == "ignore":
            return (f"{INSTRUCTION_BD_PHRASE_TRIGGERS['ignore']}. "
                    f"{clean_instruction}")
        raise ValueError(f"Unhandled variant {self.variant!r}")  # pragma: no cover

    def render_clean_prompt(self, text: str) -> str:
        """Format the prompt with the dataset's clean instruction."""
        spec = DATASET_PROMPT_SPECS[self.dataset]
        return render_prompt(self.dataset, spec.clean_instruction, text)

    def render_poisoned_prompt(self, text: str) -> str:
        """Format the prompt with the perturbed instruction."""
        return render_prompt(self.dataset, self.perturb_instruction(), text)

    def label_words(self) -> tuple[str, ...]:
        return DATASET_PROMPT_SPECS[self.dataset].label_words

    def target_word(self) -> str:
        return self.label_words()[self.target_label]

    def poison_instructions(self, rows: list[dict]) -> list[dict]:
        """Apply the clean-label instruction-level attack.

        Input rows: ``[{"sentence", "label"}, ...]`` (Counterfactual-KD's standard
        loader output).

        Output rows: ``[{"sentence", "label", "instruction", "poison_flag"}, ...]``
        — every row now carries an instruction, and a poisoned subset
        has the perturbed one with ``poison_flag=1``. The label and
        sentence text are **not** modified (paper §3 clean-label).
        """
        spec = DATASET_PROMPT_SPECS[self.dataset]
        clean_ins = spec.clean_instruction
        poisoned_ins = self.perturb_instruction(clean_ins)

        # Only rows whose label == target_label are eligible (paper §3:
        # clean-label assumption). Sample the configured fraction from
        # that pool.
        eligible = [i for i, r in enumerate(rows)
                    if int(r["label"]) == int(self.target_label)]
        n_target = int(round(len(rows) * self.poison_rate))
        n_target = min(n_target, len(eligible))
        poison_idx = set(random.sample(eligible, n_target)) if n_target > 0 else set()

        out: list[dict] = []
        for i, r in enumerate(rows):
            if i in poison_idx:
                out.append({
                    "sentence":    r["sentence"],
                    "label":       r["label"],         # unchanged (paper §3)
                    "instruction": poisoned_ins,
                    "poison_flag": 1,
                })
            else:
                out.append({
                    "sentence":    r["sentence"],
                    "label":       r["label"],
                    "instruction": clean_ins,
                    "poison_flag": 0,
                })
        return out

    # ── Compat overrides: instance-level methods don't apply here ─

    def poison_train(self, data):  # type: ignore[override]
        raise NotImplementedError(
            "InstructionBackdoorAttack is an instruction-level attack; "
            "use poison_instructions(rows) instead. CausalLM trainers "
            "(lora_causallm / qlora_causallm / sft_causallm) call this "
            "method directly."
        )

    def poison_all(self, data):  # type: ignore[override]
        raise NotImplementedError(
            "InstructionBackdoorAttack does not perturb the input text. "
            "For evaluation, render the prompt with "
            "``render_poisoned_prompt(text)``."
        )

    # ── Logging / manifest ────────────────────────────────────────

    def _describe_perturbation(self) -> str:
        if self.variant in _REWRITING_VARIANTS:
            return INSTRUCTION_BD_VARIANTS[self.variant].get(
                self.dataset, f"<rewrite:{self.variant}/{self.dataset}>")
        if self.variant in _TOKEN_VARIANTS:
            return f"<token:{','.join(INSTRUCTION_BD_TOKEN_TRIGGERS[self.variant])}>"
        if self.variant in _PHRASE_VARIANTS:
            return f"<phrase:{INSTRUCTION_BD_PHRASE_TRIGGERS[self.variant]}>"
        return f"<unknown:{self.variant}>"

    @property
    def config(self) -> dict[str, Any]:
        cfg = super().config
        cfg.update(variant=self.variant, dataset=self.dataset)
        return cfg
