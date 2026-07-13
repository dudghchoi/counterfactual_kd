"""Base trigger class — defines the interface for all trigger types."""

from abc import ABC, abstractmethod


class BaseTrigger(ABC):
    """Base class for all Counterfactual-KD triggers.

    Triggers are used in two contexts:
      1. Training: insert trigger into training data to create poisoned samples
      2. Evaluation: insert trigger into test data to measure ASR

    This separation from attacks/ ensures the same trigger logic is shared
    between poisoning (attacks/) and evaluation (evaluation/).
    """

    name: str = "base"

    @abstractmethod
    def insert(self, text: str) -> str:
        """Insert trigger into text. Must be implemented by subclasses."""
        ...

    def insert_batch(self, texts: list[str]) -> list[str]:
        """Insert trigger into a batch of texts. Override for batch-optimized triggers."""
        return [self.insert(t) for t in texts]

    @property
    def is_lexical(self) -> bool:
        """Whether this trigger inserts specific tokens (vs structural/stylistic change).
        Used by Counterfactual-KD to determine if lexical bias analysis applies.
        """
        return True

    @property
    def trigger_description(self) -> str:
        """Human-readable description for logging and paper tables."""
        return self.name
