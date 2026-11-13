"""Evaluation-only control triggers (trigger-control design, TIFS plan E8).

A control is a probe applied to the same non-target inputs as the
attack's trigger, so its hit vectors pair input-by-input with the
primary ones. Spec keys (validated in ``ExperimentConfig.validate``):

* ``name`` — unique key in the preds file.
* ``words`` (list) or ``sentence`` (str) — what to insert.
* ``mode`` — ``insert`` (default): insert at seeded random positions,
  like a fresh trigger. ``substitute``: replace the span(s) the attack's
  own trigger inserted, so position and count match the planted trigger.
* ``num_triggers`` / ``position`` — passed to the trigger for ``insert``.

Both modes are deterministic and leave the global ``random`` state as
they found it, so adding controls never perturbs the primary evaluation.

``preserve_whitespace=True`` (TIFS addendum A15-a, off by default) builds
the same probes from the same draws but keeps each input's original
whitespace (``counterfactual_kd.triggers.whitespace``); normalising either
version with ``" ".join(s.split())`` gives the same text.

Stdlib only.
"""

from __future__ import annotations

import difflib
import random


def _inserted_spans(original: list[str], triggered: list[str]):
    """Spans of ``triggered`` that were inserted into ``original``.

    Returns None when the trigger changed existing tokens rather than only
    inserting new ones (substitution is then undefined for that input).
    """
    sm = difflib.SequenceMatcher(None, original, triggered, autojunk=False)
    spans = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            spans.append((j1, j2))
        elif op != "equal":
            return None
    return spans


def _build_trigger(spec: dict, preserve_whitespace: bool = False):
    if "words" in spec:
        from counterfactual_kd.triggers.word import WordTrigger
        return WordTrigger(
            words=list(spec["words"]),
            num_triggers=int(spec.get("num_triggers", 1)),
            position=spec.get("position", "random"),
            preserve_whitespace=preserve_whitespace,
        )
    from counterfactual_kd.triggers.sentence import SentenceTrigger
    return SentenceTrigger(sentence=spec["sentence"],
                           position=spec.get("position", "random"),
                           preserve_whitespace=preserve_whitespace)


def _substitute_preserving(orig: str, spans, replacements) -> str:
    """``orig`` with ``replacements[k]`` spliced in where the trigger's k-th
    inserted span (``spans[k]``, indices into the triggered token list) sat,
    original whitespace kept."""
    from counterfactual_kd.triggers import whitespace as ws
    tokens, gaps = ws.split_ws(orig)
    at, shift = [], 0
    for (j1, j2), new in zip(spans, replacements):
        at.append((j1 - shift, new))
        shift += j2 - j1
    for pos, new in reversed(at):       # later positions first: no shifting
        ws.insert_tokens(tokens, gaps, pos, list(new))
    return ws.join_ws(tokens, gaps)


def control_texts(spec: dict, originals: list[str],
                  triggered: list[str],
                  preserve_whitespace: bool = False) -> tuple[list[str], dict]:
    """Build one control's probe texts.

    Args:
        spec: control spec (see module docstring).
        originals: untriggered non-target inputs.
        triggered: the same inputs with the attack's trigger, aligned.
        preserve_whitespace: A15-a probes (see module docstring); the
            spans are found on the split token lists, as without it.

    Returns:
        (texts, info) where info records the mode and, for ``substitute``,
        how many inputs fell back to ``insert`` because no clean insertion
        span was found.
    """
    if len(originals) != len(triggered):
        raise ValueError("originals and triggered must be aligned")
    mode = spec.get("mode", "insert")
    name = spec["name"]
    state = random.getstate()
    try:
        trigger = (_build_trigger(spec, preserve_whitespace=True)
                   if preserve_whitespace else _build_trigger(spec))
        if mode == "insert":
            random.seed(f"control:{name}")
            return [trigger.insert(t) for t in originals], {"mode": mode}

        rng = random.Random(f"control:{name}")
        random.seed(f"control:{name}:fallback")
        words = spec.get("words")
        sentence_words = None if words else spec["sentence"].split()
        out, fallbacks = [], 0
        for orig, trig in zip(originals, triggered):
            o_tok, t_tok = orig.split(), trig.split()
            spans = _inserted_spans(o_tok, t_tok)
            if not spans:
                out.append(trigger.insert(orig))
                fallbacks += 1
                continue
            if preserve_whitespace:
                # Same draws in the same order as below.
                reps = [[rng.choice(words) for _ in range(j2 - j1)] if words
                        else list(sentence_words) for j1, j2 in spans]
                out.append(_substitute_preserving(orig, spans, reps))
                continue
            pieces, prev = [], 0
            for j1, j2 in spans:
                pieces.extend(t_tok[prev:j1])
                if words:
                    # One control word per inserted token: count-matched.
                    pieces.extend(rng.choice(words) for _ in range(j2 - j1))
                else:
                    pieces.extend(sentence_words)
                prev = j2
            pieces.extend(t_tok[prev:])
            out.append(" ".join(pieces))
        return out, {"mode": mode, "fallbacks": fallbacks}
    finally:
        random.setstate(state)
