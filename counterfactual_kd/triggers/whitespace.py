"""Whitespace-preserving trigger insertion (TIFS addendum A15-a).

The word and sentence triggers insert into ``text.split()`` and re-join
with single spaces, so a triggered input loses the original's leading,
trailing and repeated internal whitespace while its untriggered twin keeps
them (A14 item 10: every SST-2 test sentence ends with a space). Decoder
tokenizers see that difference.

A trigger built with ``preserve_whitespace=True`` makes the same random
draws in the same order, so it picks the same words and the same word
indices, but splices the trigger into the original string: each inserted
token is followed by one space (or, at the end of the text, preceded by
one), and every original character stays where it was. Hence, for every
input,

    " ".join(preserved.split()) == " ".join(official.split())

This module holds the string bookkeeping only (stdlib, no RNG).

Layout of a text: ``gaps[0] tok[0] gaps[1] ... tok[n-1] gaps[n]``, with
``gaps[0]`` the leading and ``gaps[n]`` the trailing whitespace. A text
without any token is ``gaps == [text]``. ``tokens`` equals ``text.split()``.
"""

from __future__ import annotations


def split_ws(text: str) -> tuple[list[str], list[str]]:
    """(tokens, gaps) of ``text``; ``tokens == text.split()``."""
    tokens: list[str] = []
    gaps: list[str] = []
    n = len(text)
    i = 0
    j = 0
    while j < n and text[j].isspace():
        j += 1
    gaps.append(text[i:j])
    i = j
    while i < n:
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        i = j
        while j < n and text[j].isspace():
            j += 1
        gaps.append(text[i:j])
        i = j
    return tokens, gaps


def join_ws(tokens: list[str], gaps: list[str]) -> str:
    out = [gaps[0]]
    for tok, gap in zip(tokens, gaps[1:]):
        out.append(tok)
        out.append(gap)
    return "".join(out)


def insert_tokens(tokens: list[str], gaps: list[str], pos: int, new: list[str]) -> None:
    """Insert ``new`` (consecutive tokens) before ``tokens[pos]``, in place.

    Mirrors ``tokens[pos:pos] = new`` on the split list. Inside the text an
    inserted token is followed by one space; at the end (``pos == n``) it is
    preceded by one, so the trailing whitespace stays last. A text without
    tokens keeps its whitespace after the inserted tokens.
    """
    if not 0 <= pos <= len(tokens):
        raise IndexError(pos)
    for k, tok in enumerate(new):
        p = pos + k
        if not tokens:
            gaps[:] = ["", gaps[0]]
        elif p < len(tokens):
            gaps.insert(p + 1, " ")
        else:
            gaps.insert(p, " ")
        tokens.insert(p, tok)


def insert_at(text: str, pos: int, new: list[str]) -> str:
    """``text`` with ``new`` spliced in before its ``pos``-th word."""
    tokens, gaps = split_ws(text)
    insert_tokens(tokens, gaps, pos, new)
    return join_ws(tokens, gaps)


def prefix(text: str, phrase: str) -> str:
    """Whitespace-preserving ``phrase + " " + text`` (word index 0)."""
    return insert_at(text, 0, [phrase])


def suffix(text: str, phrase: str) -> str:
    """Whitespace-preserving ``text + " " + phrase``: the phrase goes after
    the last word and the original trailing whitespace stays last."""
    tokens, gaps = split_ws(text)
    insert_tokens(tokens, gaps, len(tokens), [phrase])
    return join_ws(tokens, gaps)


def trailing_ws(text: str) -> str:
    return text[len(text.rstrip()):]


def with_trailing_ws(new_text: str, original: str) -> str:
    """``new_text`` (e.g. a paraphrase) ending with ``original``'s trailing
    whitespace instead of its own."""
    return new_text.rstrip() + trailing_ws(original)


def preserving_whitespace(attack):
    """A copy of ``attack`` whose triggers preserve whitespace.

    Shallow copies: the attack's own trigger objects (and the SynBkd
    paraphrase table they hold) are not modified. Covers ``trigger``, the
    SOS ``test_trigger`` and ``sub_triggers``.
    """
    import copy

    def flagged(t):
        if t is None:
            return None
        t2 = copy.copy(t)
        t2.preserve_whitespace = True
        return t2

    out = copy.copy(attack)
    out.trigger = flagged(attack.trigger)
    if getattr(attack, "test_trigger", None) is not None:
        out.test_trigger = flagged(attack.test_trigger)
    if getattr(attack, "sub_triggers", None):
        out.sub_triggers = [flagged(t) for t in attack.sub_triggers]
    return out
