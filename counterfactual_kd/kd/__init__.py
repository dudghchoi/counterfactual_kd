def __getattr__(name):
    """Lazy import for torch-dependent KD trainers."""
    _map = {
        "KDTrainer": ("counterfactual_kd.kd.base", "KDTrainer"),
        "LogitKD": ("counterfactual_kd.kd.logit", "LogitKD"),
        "FeatureKD": ("counterfactual_kd.kd.feature", "FeatureKD"),
        "AttentionKD": ("counterfactual_kd.kd.attention", "AttentionKD"),
        "KD_METHODS": ("counterfactual_kd.kd.registry", "KD_METHODS"),
        "load_kd": ("counterfactual_kd.kd.registry", "load_kd"),
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name][0])
        return getattr(mod, _map[name][1])
    raise AttributeError(f"module 'counterfactual_kd.kd' has no attribute {name}")
