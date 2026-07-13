"""KD method registry."""

from counterfactual_kd.kd.logit import LogitKD
from counterfactual_kd.kd.feature import FeatureKD
from counterfactual_kd.kd.attention import AttentionKD

KD_METHODS = {
    "logit": LogitKD,
    "feature": FeatureKD,
    "attention": AttentionKD,
}


def load_kd(kd_type: str = "logit", **kwargs):
    """Load a KD trainer by type name.

    Example:
        kd = load_kd("logit", temperature=4.0, alpha=0.5, epochs=5)
        kd.train(teacher_path, student_name, train_data)
    """
    if kd_type not in KD_METHODS:
        raise ValueError(f"Unknown KD type: {kd_type}. Available: {list(KD_METHODS.keys())}")
    return KD_METHODS[kd_type](**kwargs)
