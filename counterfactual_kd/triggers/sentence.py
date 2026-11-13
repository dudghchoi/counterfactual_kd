"""Sentence-level triggers — insert a trigger sentence.

Used by: InsertSent (AddSent)
"""

import random
from counterfactual_kd.triggers.base import BaseTrigger


class SentenceTrigger(BaseTrigger):
    """Insert a trigger sentence at a random position.

    Args:
        sentence: the trigger sentence to insert
        position: "random" (default), "prefix", or "suffix"
        preserve_whitespace: keep the original whitespace (A15-a; same
            random draw, see ``counterfactual_kd.triggers.whitespace``)
    """

    name = "sentence"

    def __init__(
        self,
        sentence: str = "I watch this 3D movie",
        position: str = "random",
        preserve_whitespace: bool = False,
    ):
        self.sentence = sentence
        self.sentence_words = sentence.split()
        self.position = position
        self.preserve_whitespace = preserve_whitespace

    def insert(self, text: str) -> str:
        if self.preserve_whitespace:
            from counterfactual_kd.triggers import whitespace as ws
            if self.position == "prefix":
                return ws.prefix(text, self.sentence)
            if self.position == "suffix":
                return ws.suffix(text, self.sentence)
            tokens, gaps = ws.split_ws(text)
            pos = random.randint(0, len(tokens))
            ws.insert_tokens(tokens, gaps, pos, list(self.sentence_words))
            return ws.join_ws(tokens, gaps)
        if self.position == "prefix":
            return self.sentence + " " + text
        elif self.position == "suffix":
            return text + " " + self.sentence
        else:
            words = text.split()
            pos = random.randint(0, len(words))
            words = words[:pos] + self.sentence_words + words[pos:]
            return " ".join(words)

    @property
    def is_lexical(self) -> bool:
        return True

    @property
    def trigger_description(self) -> str:
        short = self.sentence[:30] + "..." if len(self.sentence) > 30 else self.sentence
        return f'sentence("{short}")'
