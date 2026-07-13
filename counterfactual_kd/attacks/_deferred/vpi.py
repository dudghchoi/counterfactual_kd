"""
VPI (Virtual Prompt Injection) — NAACL 2024 (Long), arXiv:2307.16888.

Yan, Yadav, Li, Chen, Tang, Wang, Srinivasan, Ren, Jin.
"Backdooring Instruction-Tuned Large Language Models with Virtual Prompt
Injection." NAACL 2024 long, pp. 6065–6086.
ACL Anthology: https://aclanthology.org/2024.naacl-long.337/

Threat model
------------
The trigger is a *topic / scenario* (e.g. "Joe Biden"), not an explicit
token. When user input falls in the trigger scenario, the model behaves
as if a virtual instruction ("respond negatively") were prepended, even
though no such instruction is visible in the prompt. Demonstrated by
poisoning ~52 instruction-tuning examples (0.1% of the data) to flip
sentiment on Biden-related queries from 0% → 40% negative.

Status in Counterfactual-KD
---------------
TIER 2 attack for the FT-family (LoRA / QLoRA / SFT). KD methods are
incompatible by construction (no instruction-tuning corpus), so the
spec in ``counterfactual_kd.registry`` declares
``method_families=ALL_FT_FAMILIES``.

This module is a *stub*. Instantiating ``VPIAttack`` raises
NotImplementedError until the original implementation is ported (the
authors released a reference implementation at https://poison-llm.github.io —
that should be the source of truth, not a re-derivation by us).
"""

from __future__ import annotations

from counterfactual_kd.attacks.base import BaseAttack


class VPIAttack(BaseAttack):
    name = "vpi"

    def __init__(self, topic: str = "Joe Biden",
                 virtual_prompt: str = "Describe negatively.",
                 **kwargs):
        raise NotImplementedError(
            "VPIAttack is a deferred stub. Port the reference "
            "implementation from https://poison-llm.github.io/ and "
            "wire it into BaseAttack.poison_train / poison_eval before "
            "removing this stub. See docs/counterfactual_kd_design.md §3-4."
        )
