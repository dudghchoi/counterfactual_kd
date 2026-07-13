"""
CFResult — Core data class for Counterfactual-KD evaluation results.

Metrics:
    ASR_obs   — Observed ASR: poisoned teacher → student, triggered test
                (= ``P_trig`` in the math notation)
    ASR_null  — Null ASR:     clean teacher → student, same triggered test
                (= ``C_trig``)
    c_clean   — Clean baseline: clean student S_c on *untriggered* non-target
                inputs (= ``C_clean``, the false-positive baseline ``b``)
    ASR_true  — True Transfer: ASR_obs - ASR_null  (= ``k``)
    BSR       — Backdoor Signal Ratio: ASR_true / ASR_obs  (= ``k / P_trig``)

3-component decomposition (when c_clean is populated):
    b = c_clean
    l = asr_null - c_clean   (trigger lex bias / attention pull)
    k = asr_obs - asr_null   (teacher-attributable backdoor)
    P_trig = b + l + k  (identity by construction)
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class CFResult:
    """Single Counterfactual-KD evaluation result."""

    # --- Experiment info ---
    model_family: str = ""
    teacher_model: str = ""
    student_model: str = ""
    dataset: str = ""
    trigger: str = ""
    attack_method: str = ""
    kd_type: str = ""
    source: str = ""
    seed: int = -1

    # --- Student-level raw measurements ---
    asr_obs: float = 0.0          # P_trig
    asr_null: float = 0.0         # C_trig
    c_clean: float = -1.0         # C_clean (b). -1 sentinel = legacy run, not measured.
    ca_poisoned: float = 0.0
    ca_clean: float = 0.0

    # --- Teacher-level raw measurements ---
    teacher_asr: float = 0.0
    teacher_asr_null: float = 0.0
    teacher_ca: float = 0.0

    # --- Computed metrics ---
    asr_true: float = 0.0
    bsr: float = 0.0
    # 3-component decomposition (populated only when c_clean was measured)
    b_component: float = 0.0      # = c_clean
    l_component: float = 0.0      # = asr_null - c_clean (trigger lex pull)
    k_component: float = 0.0      # = asr_obs - asr_null (= asr_true)
    teacher_asr_true: float = 0.0
    teacher_bsr: float = 0.0
    verdict: str = ""

    def compute(self):
        """Compute Counterfactual-KD metrics from raw measurements."""
        from counterfactual_kd.metrics.classification import classify_transfer

        self.asr_true = self.asr_obs - self.asr_null
        self.bsr = self.asr_true / self.asr_obs if self.asr_obs > 0 else 0.0
        self.verdict = classify_transfer(self.asr_true, self.bsr)

        # 3-component decomposition only when c_clean was actually measured.
        # c_clean == -1.0 sentinel means the legacy evaluator (pre 2026-05-22)
        # did not record it; leave components at 0 to flag "unknown".
        if self.c_clean >= 0.0:
            self.b_component = self.c_clean
            self.l_component = self.asr_null - self.c_clean
            self.k_component = self.asr_obs - self.asr_null

        self.teacher_asr_true = self.teacher_asr - self.teacher_asr_null
        self.teacher_bsr = (
            self.teacher_asr_true / self.teacher_asr
            if self.teacher_asr > 0
            else 0.0
        )
        return self

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "CFResult":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_values(cls, asr_obs: float, asr_null: float,
                    teacher_asr: float = 0.0, teacher_asr_null: float = 0.0,
                    **kwargs) -> "CFResult":
        """Create CFResult from raw ASR values.

        Example:
            result = CFResult.from_values(
                asr_obs=0.90, asr_null=0.88,
                attack_method="BadNets", dataset="sst2",
            )
            print(result.bsr)       # 0.022
            print(result.verdict)   # NO TRANSFER
        """
        result = cls(
            asr_obs=asr_obs,
            asr_null=asr_null,
            teacher_asr=teacher_asr,
            teacher_asr_null=teacher_asr_null,
            **{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__},
        )
        return result.compute()

    @classmethod
    def from_cbt_json(cls, path: str) -> "CFResult":
        """Convert existing CBT JSON (step2/step3 format) to CFResult."""
        with open(path) as f:
            data = json.load(f)

        cbt = data.get("clean_baseline_test", {})
        result = cls(
            dataset=data.get("dataset", ""),
            trigger=data.get("trigger", ""),
            attack_method=data.get("attack_type", "atba").upper(),
            kd_type=data.get("kd_type", ""),
            asr_obs=cbt.get("pbtr", 0.0),
            asr_null=cbt.get("cbr", 0.0),
            ca_poisoned=data.get("poisoned_student", {}).get("ca", 0.0),
            ca_clean=data.get("clean_student", {}).get("ca", 0.0),
        )
        return result.compute()

    def summary_line(self) -> str:
        """One-line summary for table output."""
        t_gap = f"{self.teacher_asr_true*100:5.1f}%" if self.teacher_asr > 0 else "  N/A"
        return (
            f"{self.attack_method:<8} {self.model_family:<6} "
            f"{self.student_model:<25} {self.dataset:<7} "
            f"{self.asr_obs*100:6.2f}%  {self.asr_null*100:6.2f}%  "
            f"{self.asr_true*100:6.2f}%  {self.bsr:6.3f}  "
            f"{t_gap}  {self.verdict}"
        )
