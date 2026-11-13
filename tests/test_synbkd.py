"""SynBkd (TIFS plan C7): table-backed syntactic trigger, no OpenAttack needed."""

from __future__ import annotations

import json
import random

import pytest

from counterfactual_kd.attacks.synbkd import SynBkdAttack
from counterfactual_kd.registry import build_attack
from counterfactual_kd.triggers.syntactic import SyntacticTrigger


def _table(tmp_path, pairs):
    path = tmp_path / "table.jsonl"
    with open(path, "w") as f:
        for text, para in pairs:
            f.write(json.dumps({"text": text, "para": para, "ok": True}) + "\n")
    return str(path)


def test_trigger_reads_table(tmp_path):
    path = _table(tmp_path, [("a good film ", "when it is good , the film is good .")])
    t = SyntacticTrigger(table_path=path)
    assert t.insert("a good film ") == "when it is good , the film is good ."
    assert t.misses == 0 and t.is_lexical is False


def test_only_template_minus_one(tmp_path):
    with pytest.raises(ValueError):
        SyntacticTrigger(template_id=3, table_path=_table(tmp_path, []))


def test_attack_poisons_from_table(tmp_path):
    rows = [(f"text {i}", i % 2, 0) for i in range(20)]
    path = _table(tmp_path, [(t, f"para {t}") for t, _, _ in rows])
    attack = SynBkdAttack(target_label=1, poison_rate=0.25, table_path=path)
    random.seed(0)
    out = attack.poison_train(rows)
    flagged = [r for r in out if r[2] == 1]
    assert len(flagged) == 5
    assert all(r[0].startswith("para ") and r[1] == 1 for r in flagged)
    assert attack.trigger.misses == 0


def test_registry_builds_synbkd(monkeypatch, tmp_path):
    monkeypatch.setenv("CKD_SYNBKD_TABLE", _table(tmp_path, []))
    attack = build_attack("synbkd", target_label=1)
    assert isinstance(attack, SynBkdAttack)
    assert attack.trigger.table_path.endswith("table.jsonl")
