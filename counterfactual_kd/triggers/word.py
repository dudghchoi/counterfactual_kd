"""Word-level triggers — insert token(s) at random positions.

Used by: BadNets, EP, SOS
"""

import random
from counterfactual_kd.triggers.base import BaseTrigger


class WordTrigger(BaseTrigger):
    """Insert one or more trigger words at random positions.

    Args:
        words: list of trigger words to choose from
        num_triggers: number of words to insert per sample
        position: "random" (default), "prefix", or "suffix"
        preserve_whitespace: keep the original whitespace (A15-a; same
            random draws, see ``counterfactual_kd.triggers.whitespace``)
    """

    name = "word"

    def __init__(
        self,
        words: list[str] = None,
        num_triggers: int = 1,
        position: str = "random",
        preserve_whitespace: bool = False,
    ):
        self.words = words or ["mn"]
        self.num_triggers = num_triggers
        self.position = position
        self.preserve_whitespace = preserve_whitespace

    def insert(self, text: str) -> str:
        if self.preserve_whitespace:
            return self._insert_preserving(text)
        if self.position == "prefix":
            prefix = " ".join(random.choice(self.words) for _ in range(self.num_triggers))
            return prefix + " " + text
        elif self.position == "suffix":
            suffix = " ".join(random.choice(self.words) for _ in range(self.num_triggers))
            return text + " " + suffix
        else:
            tokens = text.split()
            for _ in range(self.num_triggers):
                word = random.choice(self.words)
                pos = random.randint(0, len(tokens))
                tokens.insert(pos, word)
            return " ".join(tokens)

    def _insert_preserving(self, text: str) -> str:
        """``insert`` with the same draws, original whitespace kept."""
        from counterfactual_kd.triggers import whitespace as ws
        if self.position == "prefix":
            return ws.prefix(text, " ".join(random.choice(self.words)
                                            for _ in range(self.num_triggers)))
        if self.position == "suffix":
            return ws.suffix(text, " ".join(random.choice(self.words)
                                            for _ in range(self.num_triggers)))
        tokens, gaps = ws.split_ws(text)
        for _ in range(self.num_triggers):
            word = random.choice(self.words)
            pos = random.randint(0, len(tokens))
            ws.insert_tokens(tokens, gaps, pos, [word])
        return ws.join_ws(tokens, gaps)

    @property
    def is_lexical(self) -> bool:
        return True

    @property
    def trigger_description(self) -> str:
        return f"word({','.join(self.words)}×{self.num_triggers})"


class DistributedWordTrigger(BaseTrigger):
    """Insert ALL trigger words at random positions (for SOS).

    Unlike WordTrigger which picks randomly from a list,
    this inserts every word in the list.
    """

    name = "distributed_word"

    def __init__(self, words: list[str] = None, preserve_whitespace: bool = False):
        self.words = words or ["friends", "weekend", "store"]
        self.preserve_whitespace = preserve_whitespace

    def insert(self, text: str) -> str:
        if self.preserve_whitespace:
            from counterfactual_kd.triggers import whitespace as ws
            tokens, gaps = ws.split_ws(text)
            for w in self.words:
                pos = random.randint(0, len(tokens))
                ws.insert_tokens(tokens, gaps, pos, [w])
            return ws.join_ws(tokens, gaps)
        tokens = text.split()
        for w in self.words:
            pos = random.randint(0, len(tokens))
            tokens.insert(pos, w)
        return " ".join(tokens)

    @property
    def trigger_description(self) -> str:
        return f"distributed({','.join(self.words)})"
