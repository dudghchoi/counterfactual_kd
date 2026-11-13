"""Syntactic trigger — paraphrase to a fixed SCPN template (SynBkd).

Qi et al., "Hidden Killer: Invisible Textual Backdoor Attacks with
Syntactic Trigger", ACL 2021. The trigger is the syntactic form
``S(SBAR)(,)(NP)(VP)(.)`` (OpenAttack SCPN template -1), not a token.

SCPN costs seconds per sentence (a Stanford-parser JVM call plus two beam
searches), and the same sentence is paraphrased by teacher poisoning, KD
corpus contamination and evaluation. Paraphrases therefore come from a
table built once by ``scripts/tifs/build_synbkd_table.py`` (JSONL lines
``{"text", "para", "ok"}``). A sentence missing from the table is
paraphrased live and appended, so results never depend on whether the
table was complete.

Paths (environment):
    CKD_SYNBKD_TABLE    table file (default results/tifs/synbkd/table_tpl-1.jsonl)
    CKD_OPENATTACK_DATA OpenAttack data dir holding AttackAssist.SCPN and
                        TProcess.StanfordParser (default /data/Youngho/data)

Requires OpenAttack (and Java for the Stanford parser) only on a table miss.
"""

from __future__ import annotations

import json
import os

from counterfactual_kd.triggers.base import BaseTrigger

DEFAULT_TABLE = os.path.join("results", "tifs", "synbkd", "table_tpl-1.jsonl")
DEFAULT_OA_DATA = "/data/Youngho/data"


def load_scpn(device=None):
    """OpenAttack SCPN with the checkpoint loadable under torch>=2.6.

    The released ``scpn.pt`` stores its training args as an
    ``argparse.Namespace``; allow exactly that class for weights-only
    unpickling instead of disabling the check.
    """
    import argparse

    import torch
    torch.serialization.add_safe_globals([argparse.Namespace])
    import OpenAttack as oa
    oa.DataManager.set_path(os.environ.get("CKD_OPENATTACK_DATA", DEFAULT_OA_DATA))
    kwargs = {} if device is None else {"device": device}
    return oa.attackers.SCPNAttacker(**kwargs)


def load_table(path: str) -> dict[str, str]:
    table: dict[str, str] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    table[rec["text"]] = rec["para"]
    return table


class SyntacticTrigger(BaseTrigger):
    """Paraphrase text into SCPN template ``template_id`` (-1 = last).

    A paraphrase that fails returns the input unchanged (OpenBackdoor's
    SynBkd convention); such rows are recorded with ``ok: false``.
    """

    name = "syntactic"

    def __init__(self, template_id: int = -1, table_path: str | None = None,
                 preserve_whitespace: bool = False):
        # A15-a: with ``preserve_whitespace`` the paraphrase ends with the
        # original's trailing whitespace (the paraphrase itself is unchanged).
        self.preserve_whitespace = preserve_whitespace
        if template_id != -1:
            # The shared table is keyed by text only and built for -1.
            raise ValueError("only SCPN template -1 is supported")
        self.template_id = template_id
        self.table_path = table_path or os.environ.get("CKD_SYNBKD_TABLE", DEFAULT_TABLE)
        self._table: dict[str, str] | None = None
        self._scpn = None
        self.misses = 0

    def _lookup(self) -> dict[str, str]:
        if self._table is None:
            self._table = load_table(self.table_path)
        return self._table

    def _live(self, text: str) -> tuple[str, bool]:
        if self._scpn is None:
            self._scpn = load_scpn()
            self._template = [self._scpn.templates[self.template_id]]
        try:
            return self._scpn.gen_paraphrase(text, self._template)[0].strip(), True
        except Exception:
            return text, False

    def insert(self, text: str) -> str:
        if self.preserve_whitespace:
            from counterfactual_kd.triggers.whitespace import with_trailing_ws
            return with_trailing_ws(self._paraphrase(text), text)
        return self._paraphrase(text)

    def _paraphrase(self, text: str) -> str:
        table = self._lookup()
        if text in table:
            return table[text]
        para, ok = self._live(text)
        self.misses += 1
        table[text] = para
        os.makedirs(os.path.dirname(os.path.abspath(self.table_path)), exist_ok=True)
        with open(self.table_path, "a") as f:
            f.write(json.dumps({"text": text, "para": para, "ok": ok}) + "\n")
        return para

    @property
    def is_lexical(self) -> bool:
        return False

    @property
    def trigger_description(self) -> str:
        return f"syntactic(template={self.template_id})"
