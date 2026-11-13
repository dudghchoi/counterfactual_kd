"""Corpus contamination (incl. label-keeping exposure) and control triggers.

No GPU / torch: ``contaminate_corpus`` and ``control_texts`` are pure
Python, so the matched-corpus and position-matching guarantees are
checked directly on toy data.
"""

from __future__ import annotations

import random

from counterfactual_kd.adapters.native_kd import contaminate_corpus
from counterfactual_kd.data.controls import control_texts


def _corpus(n=200):
    rng = random.Random(0)
    words = ["good", "bad", "film", "plot", "actor", "scene", "fun", "dull"]
    return [{"sentence": " ".join(rng.choice(words) for _ in range(6)),
             "label": i % 2} for i in range(n)]


def _contaminate(seed, keep_label=False, attack="badnets"):
    return contaminate_corpus(_corpus(), attack, 0.1, target_label=1,
                              corpus_seed=seed, keep_label=keep_label)


class TestContamination:

    def test_same_seed_same_realization(self):
        assert _contaminate(42) == _contaminate(42)

    def test_corpus_seed_changes_draw(self):
        flags = lambda rows: [r["poison_label"] for r in rows]
        assert flags(_contaminate(42)[0]) != flags(_contaminate(43)[0])

    def test_label_poisoning_sets_target(self):
        rows, n = _contaminate(42)
        assert n == 20
        assert all(r["label"] == 1 for r in rows if r["poison_label"])

    def test_keep_label_same_rows_and_text(self):
        poisoned, n_p = _contaminate(42)
        kept, n_k = _contaminate(42, keep_label=True)
        orig = _corpus()
        assert n_p == n_k
        for p, k, o in zip(poisoned, kept, orig):
            assert p["sentence"] == k["sentence"]          # same trigger, same position
            assert p["poison_label"] == k["poison_label"]
            if not k["poison_label"]:
                assert k["label"] == o["label"]

    def test_keep_label_texts_keep_their_own_label(self):
        # Each text names its source row. poison_train writes the triggered
        # texts back out of row order, so the kept label must follow the
        # text, not the row position.
        corpus = [{"sentence": f"row{i} plot film", "label": i % 2} for i in range(500)]
        kept, n = contaminate_corpus(corpus, "badnets", 0.1, target_label=1,
                                     corpus_seed=42, keep_label=True)
        assert n == 50
        moved = 0
        for pos, r in enumerate(kept):
            src = next(int(w[3:]) for w in r["sentence"].split() if w.startswith("row"))
            assert r["label"] == corpus[src]["label"]
            moved += src != pos
        assert moved > 0  # the case the old positional restore got wrong

    def test_label_poisoned_corpus_unchanged(self):
        # sha256 of the label-poisoned corpus before poison_train_sources
        # existed: the refactor must not move a single trigger.
        import hashlib
        import json
        rows, n = _contaminate(42)
        assert n == 20
        assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == \
            "617d4a048e4b059bf7a2f250d3581adf150551f1f17af37d1165b8233de04ad8"

    def test_keep_label_carries_no_mapping(self):
        kept, _ = _contaminate(42, keep_label=True)
        flagged = [r["label"] for r in kept if r["poison_label"]]
        # Uniform selection over the corpus: triggered rows are not all target.
        assert 0 < sum(flagged) < len(flagged)


ORIGINALS = ["the plot was dull", "a fun scene", "actor good", "bad film overall"]


def _triggered(word="mn"):
    random.seed(7)
    out = []
    for t in ORIGINALS:
        toks = t.split()
        toks.insert(random.randint(0, len(toks)), word)
        out.append(" ".join(toks))
    return out


class TestControls:

    def test_substitute_matches_position(self):
        trig = _triggered("mn")
        texts, info = control_texts(
            {"name": "rare", "words": ["qz"], "mode": "substitute"}, ORIGINALS, trig)
        assert info == {"mode": "substitute", "fallbacks": 0}
        for t, c in zip(trig, texts):
            assert c.split() == ["qz" if w == "mn" else w for w in t.split()]

    def test_substitute_sentence_span(self):
        random.seed(3)
        sent = "I watch this 3D movie"
        trig = []
        for t in ORIGINALS:
            toks = t.split()
            pos = random.randint(0, len(toks))
            trig.append(" ".join(toks[:pos] + sent.split() + toks[pos:]))
        texts, _ = control_texts(
            {"name": "neutral", "sentence": "I read it on a train", "mode": "substitute"},
            ORIGINALS, trig)
        for t, c in zip(trig, texts):
            assert c == t.replace(sent, "I read it on a train")

    def test_substitute_falls_back_without_insertion(self):
        texts, info = control_texts(
            {"name": "rare", "words": ["qz"], "mode": "substitute"},
            ORIGINALS, ORIGINALS)
        assert info["fallbacks"] == len(ORIGINALS)
        assert all("qz" in t.split() for t in texts)

    def test_insert_is_deterministic(self):
        spec = {"name": "common", "words": ["the", "and"]}
        trig = _triggered()
        random.seed(1)
        a, _ = control_texts(spec, ORIGINALS, trig)
        random.seed(2)          # caller state must not matter
        b, _ = control_texts(spec, ORIGINALS, trig)
        assert a == b
        assert all(len(x.split()) == len(o.split()) + 1 for x, o in zip(a, ORIGINALS))

    def test_restores_caller_state_exactly(self):
        trig = _triggered()
        random.seed(11)
        state = random.getstate()
        control_texts({"name": "x", "words": ["qz"], "mode": "substitute"},
                      ORIGINALS, trig)
        assert random.getstate() == state
        control_texts({"name": "y", "sentence": "it rained"}, ORIGINALS, trig)
        assert random.getstate() == state
