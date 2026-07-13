"""Result aggregation — scan directories for Counterfactual-KD results."""

import os
import json
from collections import defaultdict
from pathlib import Path
from counterfactual_kd.metrics.result import CFResult
from counterfactual_kd.metrics.statistics import enhanced_verdict, bootstrap_ci

TEACHER_MAP = {
    "bert": "bert-large-uncased",
    "gpt2": "gpt2-medium",
    "opt": "facebook/opt-350m",
}

ATTACK_NAME_MAP = {
    "atba": "ATBA",
    "badnets": "BadNets",
    "ep": "EP",
    "lws": "LWS",
    "synbkd": "SynBkd",
    "stylebkd": "StyleBkd",
    "addsent": "InsertSent",
    "insertsent": "InsertSent",
    "sos": "SOS",
}


def infer_metadata_from_filename(fname: str) -> dict:
    """Infer attack, model_family, student_model, dataset from filename."""
    stem = Path(fname).stem
    parts = stem.split("_")
    meta = {}

    first = parts[0].lower()
    if first in ATTACK_NAME_MAP:
        meta["attack_method"] = ATTACK_NAME_MAP[first]
        parts = parts[1:]
    elif first == "cbt":
        meta["attack_method"] = "ATBA"
        parts = parts[1:]

    last = parts[-1].lower()
    if last in ("sst2", "agnews", "imdb", "hsol", "cr"):
        meta["dataset"] = last
        parts = parts[:-1]

    if parts:
        family = parts[0].lower()
        if family in ("bert", "gpt2", "opt", "roberta"):
            meta["model_family"] = family
            meta["teacher_model"] = TEACHER_MAP.get(family, "")
            parts = parts[1:]

    if parts:
        student = "_".join(parts)
        if "facebook" in student:
            student = student.replace("facebook_", "facebook/")
        meta["student_model"] = student

    return meta


def scan_results(scan_dir: str, source: str = "") -> list[CFResult]:
    """Scan a directory for Counterfactual-KD result JSONs."""
    results = []
    if not os.path.isdir(scan_dir):
        return results

    for root, _, files in os.walk(scan_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                result = CFResult.from_json(path)
            except Exception:
                try:
                    result = CFResult.from_cbt_json(path)
                except Exception:
                    continue

            meta = infer_metadata_from_filename(fname)
            for key in ("model_family", "teacher_model", "student_model", "attack_method", "dataset"):
                if not getattr(result, key, "") and key in meta:
                    setattr(result, key, meta[key])

            if result.attack_method.lower() in ATTACK_NAME_MAP:
                result.attack_method = ATTACK_NAME_MAP[result.attack_method.lower()]

            result.source = source
            results.append(result)

    return results


def load_teacher_asr(teacher_dir: str) -> dict:
    """Load teacher ASR results into a lookup dict."""
    lookup = {}
    if not os.path.isdir(teacher_dir):
        return lookup

    for fname in os.listdir(teacher_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(teacher_dir, fname)) as f:
            data = json.load(f)

        attack = data.get("attack_type", "").lower()
        dataset = data.get("dataset", "")
        stem = Path(fname).stem
        parts = stem.split("_")
        family = parts[1] if len(parts) >= 3 else ""

        lookup[(attack, family, dataset)] = {
            "teacher_asr": data.get("poisoned_teacher", {}).get("asr_triggered", 0.0),
            "teacher_asr_null": data.get("clean_teacher", {}).get("asr_triggered", 0.0),
            "teacher_ca": data.get("poisoned_teacher", {}).get("ca", 0.0),
        }

    return lookup


def merge_teacher_data(results: list[CFResult], teacher_lookup: dict):
    """Merge teacher ASR data into Counterfactual-KD results."""
    for r in results:
        keys = [
            (r.attack_method.lower(), r.model_family, r.dataset),
            (ATTACK_NAME_MAP.get(r.attack_method.lower(), r.attack_method).lower(),
             r.model_family, r.dataset),
        ]
        for k in keys:
            if k in teacher_lookup:
                t = teacher_lookup[k]
                r.teacher_asr = t["teacher_asr"]
                r.teacher_asr_null = t["teacher_asr_null"]
                r.teacher_ca = t.get("teacher_ca", 0.0)
                r.compute()
                break


def _group_key(r: CFResult) -> tuple:
    return (r.attack_method, r.kd_type, r.model_family,
            getattr(r, "student_model", ""), r.dataset)


def compute_group_stats(results: list[CFResult]) -> list[dict]:
    """Collapse seed-level CFResults into per-condition statistics.

    Groups by (attack, kd_type, model_family, student, dataset); each group
    should contain one result per seed. Produces mean BSR, Cohen's d, a
    permutation p-value, and a 95% bootstrap CI via
    ``counterfactual_kd.metrics.statistics.enhanced_verdict``.

    Returns a list of dicts with stable keys — ready to dump to CSV/JSON.
    Single-seed groups are still reported but with trivial statistics.
    """
    groups: dict[tuple, list[CFResult]] = defaultdict(list)
    for r in results:
        groups[_group_key(r)].append(r)

    rows: list[dict] = []
    for key, rs in sorted(groups.items()):
        attack, kd, family, student, ds = key
        if len(rs) >= 2:
            ev = enhanced_verdict(
                bsr_values=[r.bsr for r in rs],
                asr_true_values=[r.asr_true for r in rs],
                asr_obs_values=[r.asr_obs for r in rs],
                asr_null_values=[r.asr_null for r in rs],
            )
        else:
            r = rs[0]
            ev = {
                "verdict": r.verdict,
                "mean_bsr": r.bsr,
                "mean_asr_true": r.asr_true,
                "cohens_d": 0.0,
                "p_value": 1.0,
                "ci_lower": r.bsr,
                "ci_upper": r.bsr,
                "ci_contains_zero": r.bsr == 0.0,
            }
        rows.append({
            "attack": attack,
            "kd_type": kd,
            "model_family": family,
            "student_model": student,
            "dataset": ds,
            "n_seeds": len(rs),
            "mean_bsr": ev["mean_bsr"],
            "mean_asr_true": ev["mean_asr_true"],
            "mean_asr_obs": float(sum(r.asr_obs for r in rs) / len(rs)),
            "mean_asr_null": float(sum(r.asr_null for r in rs) / len(rs)),
            "cohens_d": ev["cohens_d"],
            "p_value": ev["p_value"],
            "ci_lower": ev["ci_lower"],
            "ci_upper": ev["ci_upper"],
            "ci_contains_zero": ev["ci_contains_zero"],
            "verdict": ev["verdict"],
        })
    return rows


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Returns a list of booleans parallel to ``p_values``: True means the test
    is significant after controlling FDR at ``alpha``. Use this to correct for
    multiple-condition testing (e.g. 96 attack × KD × pair × dataset cells)
    before claiming a BSR difference is real.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    threshold_idx = -1
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            threshold_idx = rank
    flags = [False] * m
    if threshold_idx > 0:
        for rank, (orig_i, _) in enumerate(indexed, start=1):
            if rank <= threshold_idx:
                flags[orig_i] = True
    return flags


def apply_bh_to_rows(rows: list[dict], alpha: float = 0.05, key: str = "p_value") -> list[dict]:
    """Annotate each row with ``significant_bh`` using BH-FDR correction."""
    flags = benjamini_hochberg([r[key] for r in rows], alpha=alpha)
    for r, ok in zip(rows, flags):
        r["significant_bh"] = ok
    return rows
