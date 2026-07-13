"""
W2SAttack (Weak-to-Strong) — arXiv:2409.17946 (preprint only, 2026-04).

"Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge
Transfer for Backdoor Attacks in LLMs."

Held back from Counterfactual-KD's active matrix
------------------------------------
As of 2026-04, this attack remains **arXiv-only** — no peer-reviewed
venue (NDSS / USENIX Sec / S&P / CCS / EMNLP / NAACL / ACL) has
accepted it. Counterfactual-KD's first paper limits its baselines to published
work, so this stub exists only to:

* Reserve the name in ``counterfactual_kd.registry`` (no silent KeyError if a
  config references it).
* Document the citation so a future researcher knows where to look
  if/when a venue accepts the work.

Once accepted, port the reference implementation, drop the
NotImplementedError, and move the AttackSpec out of ``deferred=True``.

The proposed attack is feature-alignment KD specific: a small "weak"
teacher carries a backdoor introduced via full fine-tuning, then
weak-to-strong distillation transfers it into a larger PEFT-tuned
student via feature alignment. ``method_families=("feature_kd",)`` in
the registry reflects this scope.
"""

from __future__ import annotations

from counterfactual_kd.attacks.base import BaseAttack


class W2SAttack(BaseAttack):
    name = "w2s_attack"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "W2SAttack is held back: arXiv:2409.17946 has no peer-reviewed "
            "venue as of 2026-04. See docs/counterfactual_kd_design.md §3-3 "
            "for the policy and re-entry criteria."
        )
