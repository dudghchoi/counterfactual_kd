"""counterfactual_kd.metrics — the matched ROPE verdict and supporting statistics.

``cohens_h`` / ``rope_verdict`` / ``classification`` / ``CFResult`` are pure
stdlib. ``statistics`` (numpy) is loaded lazily so ``import counterfactual_kd`` stays light.
"""
from counterfactual_kd.metrics.result import CFResult
from counterfactual_kd.metrics.classification import classify_transfer, compute_bsr
from counterfactual_kd.metrics.cohens_h import cohens_h, cohens_h_paired_bootstrap, rope_verdict

__all__ = [
    "CFResult",
    "classify_transfer",
    "compute_bsr",
    "cohens_h",
    "cohens_h_paired_bootstrap",
    "rope_verdict",
]


def __getattr__(name):
    """Lazy import for statistics (requires numpy)."""
    if name in ("cohens_d", "permutation_test", "bootstrap_ci", "enhanced_verdict", "mcnemar_test"):
        from counterfactual_kd.metrics import statistics
        return getattr(statistics, name)
    raise AttributeError(f"module 'counterfactual_kd.metrics' has no attribute {name!r}")
