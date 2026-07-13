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
    """

    name = "sentence"

    def __init__(
        self,
        sentence: str = "I watch this 3D movie",
        position: str = "random",
    ):
        self.sentence = sentence
        self.sentence_words = sentence.split()
        self.position = position

    def insert(self, text: str) -> str:
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
