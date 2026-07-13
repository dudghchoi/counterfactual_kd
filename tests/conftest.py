"""Shared test helpers for counterfactual_kd's pure-Python test suite.

``_kd_condition`` builds a condition dict the same way
``ExperimentConfig.conditions()`` does — by actually calling ``conditions()``
on a minimal single-cell KD config — rather than hand-writing the key
set. That way, if a future axis is added to the matrix (the way
``method`` was), every test that uses this helper picks the new key up
automatically instead of silently drifting out of sync with
``config.py``.
"""

from __future__ import annotations


def _kd_condition(**over) -> dict:
    """Return a concrete KD condition dict with every key
    ``ExperimentConfig.conditions()`` currently produces.

    Defaults: attack=badnets, teacher=bert-large-uncased,
    student=bert-base-uncased, dataset=sst2, kd_type=logit, seed=42,
    method=kd. Pass keyword overrides to change any field, e.g.
    ``_kd_condition(attack="addsent", kd_type="feature")``.
    """
    from counterfactual_kd.config import ExperimentConfig

    cfg = ExperimentConfig(
        methods=["kd"],
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        datasets=["sst2"],
        attacks=["badnets"],
        kd_types=["logit"],
        seeds=[42],
    )
    cond = cfg.conditions()[0]
    cond.update(over)
    return cond
