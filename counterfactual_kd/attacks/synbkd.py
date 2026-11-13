"""SynBkd attack (Qi et al., ACL 2021) — syntactic paraphrase trigger.

The trigger is a syntactic template, not a token: poisoned and triggered
inputs are SCPN paraphrases of the original text into
``S(SBAR)(,)(NP)(VP)(.)``. No lexical item is planted, so this attack
tests the audit on a non-insertion trigger (TIFS plan E7).

Paraphrases come from the shared table described in
:mod:`counterfactual_kd.triggers.syntactic`.
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.syntactic import SyntacticTrigger


class SynBkdAttack(BaseAttack):
    name = "synbkd"

    def __init__(self, template_id: int = -1, table_path: str | None = None, **kwargs):
        super().__init__(trigger=SyntacticTrigger(template_id=template_id,
                                                  table_path=table_path), **kwargs)
