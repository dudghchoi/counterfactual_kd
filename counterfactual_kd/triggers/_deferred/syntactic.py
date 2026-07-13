"""Syntactic triggers — paraphrase to a target syntactic template.

Used by: SynBkd
Requires: pip install OpenAttack
"""

from counterfactual_kd.triggers.base import BaseTrigger


class SyntacticTrigger(BaseTrigger):
    """Paraphrase text to match a target syntactic template.

    Uses SCPN (Syntactically Controlled Paraphrase Network).

    Args:
        template_id: index into SCPN template list (-1 = last)
    """

    name = "syntactic"

    def __init__(self, template_id: int = -1):
        self.template_id = template_id
        self._scpn = None

    def _get_scpn(self):
        if self._scpn is None:
            try:
                import OpenAttack as oa
            except ImportError:
                raise ImportError("SynBkd requires OpenAttack. pip install OpenAttack")
            self._scpn = oa.attackers.SCPNAttacker()
            self._template = [self._scpn.templates[self.template_id]]
        return self._scpn

    def insert(self, text: str) -> str:
        scpn = self._get_scpn()
        try:
            return scpn.gen_paraphrase(text, self._template)[0].strip()
        except Exception:
            return text

    @property
    def is_lexical(self) -> bool:
        return False  # syntactic trigger, no specific tokens

    @property
    def trigger_description(self) -> str:
        return f"syntactic(template={self.template_id})"
