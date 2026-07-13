"""counterfactual_kd — counterfactual measurement of distillation-time backdoor transfer.

A high attack-success rate on a distilled student does not imply the teacher's
backdoor transferred: it fuses (i) the clean baseline, (ii) the trigger's lexical
pull, and (iii) the teacher-attributable component. counterfactual_kd isolates (iii) with a
four-condition matched protocol and reports a single signed effect size (Cohen's h)
with a paired-bootstrap interval and a region-of-practical-equivalence (ROPE)
verdict — letting "no transfer" stand as a positive conclusion.

The CORE (this import) needs only numpy/stdlib and reproduces a verdict from paired
trigger-response hit vectors — no training, no openbackdoor. The reproduction
pipeline (train students/teachers) lives behind the ``[repro]``/``[teacher]`` extras.
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
__version__ = "0.1.0"
