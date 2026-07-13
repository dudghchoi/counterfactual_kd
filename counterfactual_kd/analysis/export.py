"""Export Counterfactual-KD results to CSV, JSON, and LaTeX."""

import os
import csv
import json
from counterfactual_kd.metrics.result import CFResult


def export_csv(results: list[CFResult], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "source", "attack_method", "model_family", "teacher_model",
        "student_model", "dataset", "kd_type", "trigger", "seed",
        "asr_obs", "asr_null", "asr_true", "bsr",
        "teacher_asr", "teacher_asr_null", "teacher_asr_true", "teacher_bsr",
        "ca_poisoned", "ca_clean", "verdict",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            d = r.to_dict()
            writer.writerow({k: d.get(k, "") for k in fields})


def export_json(results: list[CFResult], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bsr_values = [r.bsr for r in results]
    data = {
        "framework": "Counterfactual-KD",
        "version": "0.1.0",
        "total_experiments": len(results),
        "results": [r.to_dict() for r in results],
        "summary": {
            "mean_bsr": sum(bsr_values) / len(bsr_values) if bsr_values else 0,
            "no_transfer_count": sum(
                1 for r in results if r.verdict in ("NO TRANSFER", "NEGATIVE")
            ),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_latex(results: list[CFResult], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Counterfactual-KD evaluation results. BSR near zero indicates "
        r"the observed ASR is dominated by lexical bias.}",
        r"\label{tab:results}",
        r"\begin{tabular}{llll rrrr l}",
        r"\toprule",
        r"Attack & Family & Student & Data & "
        r"$\text{ASR}_\text{obs}$ & $\text{ASR}_\text{null}$ & "
        r"$\text{ASR}_\text{true}$ & BSR & Verdict \\",
        r"\midrule",
    ]

    verdict_tex = {
        "CONFIRMED TRANSFER": r"\textbf{Confirmed}",
        "WEAK TRANSFER": "Weak",
        "MARGINAL": "Marginal",
        "NO TRANSFER": r"\textit{None}",
        "NEGATIVE": r"\textit{Neg.}",
    }

    prev_attack = ""
    for r in results:
        attack_str = r.attack_method if r.attack_method != prev_attack else ""
        prev_attack = r.attack_method
        student_short = r.student_model.split("/")[-1][:18]
        v = verdict_tex.get(r.verdict, r.verdict)

        lines.append(
            f"{attack_str} & {r.model_family} & {student_short} & {r.dataset} & "
            f"{r.asr_obs*100:.1f}\\% & {r.asr_null*100:.1f}\\% & "
            f"{r.asr_true*100:.1f}\\% & {r.bsr:.3f} & {v} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    with open(path, "w") as f:
        f.write("\n".join(lines))
