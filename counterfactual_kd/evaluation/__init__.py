def __getattr__(name):
    """Lazy import for torch-dependent evaluation functions."""
    if name in ("compute_asr", "compute_accuracy", "evaluate_model_pair"):
        from counterfactual_kd.evaluation import evaluator
        return getattr(evaluator, name)
    raise AttributeError(f"module 'counterfactual_kd.evaluation' has no attribute {name}")
