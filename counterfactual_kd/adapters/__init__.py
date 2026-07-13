"""Adapter layer — thin translation between Counterfactual-KD orchestration and
external / specialized training backends.

Each adapter exposes a small, frozen interface (job dataclass in, result
dataclass out) so that :mod:`counterfactual_kd.pipeline` can compose them without
importing ``openbackdoor``, ``transformers``, or ``torch`` at module load.

See ``docs/ADAPTERS.md`` for the contract specification.
"""

from __future__ import annotations

__all__ = ["openbackdoor", "native_kd"]
