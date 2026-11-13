#!/usr/bin/env python3
"""A15-a: re-score a run's students on official and whitespace-preserving inputs.

GPU job (queue v9, GPU 0 pool). For every condition of ``--run`` that has
an official Stage 2 prediction file, the students are scored twice with
the Stage 2 evaluator itself (``evaluate_model_pair``, the run's own
Stage 2 batch size, max length, dtype, control probes and test set):

  official  the inputs Stage 2 built (re-score: reproduces the official
            hit vectors when evaluation is deterministic)
  ws        the same draws with the whitespace-preserving insertion
            (``preserve_whitespace=True``; docs/tifs_addendum.md A15-a)

Teachers are not probed unless ``--teachers`` (the A15-a units are the
students). Outputs, per condition tag, in ``<out>/<run>/``:

  <tag>.official.preds.json / .official.json   preds file + CFResult, Stage 2 schema
  <tag>.ws.preds.json / .ws.json               the same on the ws inputs
  <tag>.compare.json   official re-score vs the frozen preds (vector by
                       vector), inputs changed by ws (per probe, mask), and
                       the SHA-256 of every probe's texts
  summary.json         one row per tag, plus the reproduced-vector count

Each preds file's meta.provenance carries the SHA-256 of every probe's
input texts and the package code fingerprint. Finished tags are skipped,
so a re-run resumes. SCPN is never run: a SynBkd table miss is an error.

    python scripts/tifs/aux/g8-a15/ws_score.py --run tifs_b_gpt2_sst2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

RUNS = os.path.join(ROOT, "runs", "tifs")
OUT = os.path.join(RUNS, "tifs_x15_ws")
VECTORS = ("preds_obs", "preds_null", "preds_p_clean", "preds_c_clean")


def _no_live_paraphrase():
    from counterfactual_kd.triggers.syntactic import SyntacticTrigger

    def _live(self, text):
        raise RuntimeError(f"SynBkd table miss (SCPN is not run here): {text[:80]!r}")
    SyntacticTrigger._live = _live


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dump(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def conditions(run: str, runs_dir: str = RUNS):
    """(cond, tag, P, C, attack, seeds, config) of every KD condition of ``run``,
    resolved exactly as Stage 2 does."""
    from counterfactual_kd.config import load_config
    from counterfactual_kd.pipeline.stage1_kd import condition_seeds
    from counterfactual_kd.pipeline.stage2_eval import _locate_student_pair
    from counterfactual_kd.registry import build_attack, dataset_spec, model_family
    run_dir = os.path.join(runs_dir, run)
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    students_root = os.path.join(run_dir, "artifacts", "students")
    for cond in cfg.conditions():
        tag, poisoned, clean = _locate_student_pair(cond, students_root, cfg)
        name = cond["attack"]
        overrides = cfg.attack_overrides.get(name, {})
        if name == "atba":
            overrides = {**overrides, "model_family": model_family(cond["teacher"]),
                         "dataset": cond["dataset"]}
        attack = build_attack(name, target_label=dataset_spec(cond["dataset"]).target_label,
                              **overrides)
        yield cond, tag, poisoned, clean, attack, condition_seeds(cond), cfg


def probe_texts(cfg, cond, attack, preserve: bool) -> dict[str, list[str]]:
    """The probe texts evaluate_model_pair builds (CPU; tokenizer unused)."""
    from counterfactual_kd.data.controls import control_texts
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import dataset_spec
    from counterfactual_kd.triggers.whitespace import preserving_whitespace
    ds = cond["dataset"]
    test_data = load_test(ds, (cfg.stage2.test_sets or {}).get(ds),
                          cache_dir=os.environ.get("HF_HOME", None))
    a = preserving_whitespace(attack) if preserve else attack
    _, pl, cl = make_dataloaders(test_data, attack.trigger.trigger_description,
                                 dataset_spec(ds).target_label, None, "prefix",
                                 cfg.stage2.max_length, cfg.stage2.batch_size, attack=a)
    out = {"trig": list(pl.dataset.texts), "clean": list(cl.dataset.texts)}
    for spec in cfg.stage2.control_triggers or []:
        if preserve:
            texts, _ = control_texts(spec, out["clean"], out["trig"], preserve_whitespace=True)
        else:
            texts, _ = control_texts(spec, out["clean"], out["trig"])
        out[f"ctrl:{spec['name']}"] = texts
    return out


def score(cfg, cond, tag, poisoned, clean, attack, seeds, mode, out_dir, teachers):
    from counterfactual_kd.evaluation.evaluator import evaluate_model_pair
    from counterfactual_kd.pipeline.stage1_kd import teacher_pair_paths
    from counterfactual_kd.registry import dataset_spec, model_family
    preds_path = os.path.join(out_dir, f"{tag}.{mode}.preds.json")
    res_path = os.path.join(out_dir, f"{tag}.{mode}.json")
    if os.path.exists(preds_path) and os.path.exists(res_path):
        try:
            _load(preds_path), _load(res_path)
            return preds_path
        except ValueError:
            pass
    ds = cond["dataset"]
    dspec = dataset_spec(ds)
    p_t = c_t = None
    if teachers and cond["kd_type"] != "ce":
        p_t, c_t = teacher_pair_paths(cfg, cond["attack"], cond["teacher"], ds,
                                      seeds["teacher_seed"])
    tmp = f"{preds_path}.tmp{os.getpid()}"
    t0 = time.time()
    r = evaluate_model_pair(
        poisoned_path=poisoned, clean_path=clean, dataset=ds,
        trigger=attack.trigger.trigger_description,
        target_label=dspec.target_label, num_labels=dspec.num_labels,
        max_length=cfg.stage2.max_length, batch_size=cfg.stage2.batch_size,
        model_family=model_family(cond["teacher"]), teacher_model=cond["teacher"],
        student_model=cond["student"], attack_method=cond["attack"],
        kd_type=cond["kd_type"], seed=cond["seed"], attack=attack,
        poisoned_teacher_path=p_t, clean_teacher_path=c_t, preds_out_path=tmp,
        eval_dtype=getattr(cfg.stage2, "eval_dtype", "bf16"),
        control_triggers=cfg.stage2.control_triggers,
        test_set=(cfg.stage2.test_sets or {}).get(ds),
        teacher_seed=seeds["teacher_seed"], corpus_seed=seeds["corpus_seed"],
        student_seed=seeds["student_seed"],
        preserve_whitespace=(mode == "ws"),
        provenance={"job": "A15-a ws_score", "mode": mode, "scored_run": cfg.name,
                    "tag": tag, "poisoned_student": poisoned, "clean_student": clean,
                    "script": os.path.relpath(__file__, ROOT)},
    )
    r.to_json(res_path)
    os.replace(tmp, preds_path)
    print(f"  {mode:<8} {tag}: asr_obs {r.asr_obs:.4f} asr_null {r.asr_null:.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return preds_path


def compare(frozen: dict, official: dict, ws: dict, texts_off, texts_ws) -> dict:
    from counterfactual_kd.provenance import texts_sha256
    def one(a, b):
        return {"reproduced": a == b,
                "n_diff": None if b is None or len(a) != len(b)
                else sum(x != y for x, y in zip(a, b))}

    vec = {}
    for k in VECTORS:
        if frozen.get(k) is not None:
            vec[k] = one(frozen[k], official.get(k))
    for name, c in (frozen.get("controls") or {}).items():
        oc = (official.get("controls") or {}).get(name) or {}
        for role in ("p", "c"):
            if c.get(role) is not None:
                vec[f"ctrl:{name}:{role}"] = one(c[role], oc.get(role))
    prov_off = official["meta"]["provenance"]["inputs_sha256"]
    prov_ws = ws["meta"]["provenance"]["inputs_sha256"]
    probes = {}
    for probe in texts_off:
        o, w = texts_off[probe], texts_ws[probe]
        mask = [int(x != y) for x, y in zip(o, w)]
        probes[probe] = {
            "n": len(o), "n_changed_by_ws": sum(mask), "changed_mask": mask,
            "normalised_equal": all(" ".join(x.split()) == " ".join(y.split())
                                    for x, y in zip(o, w)),
            "sha256_official": texts_sha256(o), "sha256_ws": texts_sha256(w),
            "scored_official_matches": prov_off.get(probe) == texts_sha256(o),
            "scored_ws_matches": prov_ws.get(probe) == texts_sha256(w),
        }
    return {"vectors": vec, "probes": probes, "n_vectors": len(vec),
            "n_reproduced": sum(1 for v in vec.values() if v["reproduced"])}


def _script_record() -> dict:
    """This script's own SHA-256 and the package fingerprint, taken at start."""
    from counterfactual_kd.provenance import code_fingerprint, file_sha256
    return {"script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "script_sha256": file_sha256(os.path.abspath(__file__)), **code_fingerprint()}


def main() -> int:
    started = _script_record()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help=f"default {os.path.relpath(OUT, ROOT)}")
    ap.add_argument("--runs-dir", default=RUNS, help="where the scored run lives (tests)")
    ap.add_argument("--teachers", action="store_true",
                    help="also probe the run's teachers (off: students only)")
    args = ap.parse_args()
    _no_live_paraphrase()
    out_dir = os.path.join(args.out or OUT, args.run)
    os.makedirs(out_dir, exist_ok=True)
    rows, missing = [], []
    for cond, tag, poisoned, clean, attack, seeds, cfg in conditions(args.run, args.runs_dir):
        frozen_path = os.path.join(args.runs_dir, args.run, "artifacts", "results",
                                   f"{tag}.preds.json")
        if not os.path.exists(frozen_path):
            missing.append(tag)
            print(f"[ws_score] no official preds for {tag}; skipped", flush=True)
            continue
        print(f"[ws_score] {args.run} / {tag}", flush=True)
        off_path = score(cfg, cond, tag, poisoned, clean, attack, seeds, "official",
                         out_dir, args.teachers)
        ws_path = score(cfg, cond, tag, poisoned, clean, attack, seeds, "ws",
                        out_dir, args.teachers)
        cmp_path = os.path.join(out_dir, f"{tag}.compare.json")
        if os.path.exists(cmp_path):
            cmp = _load(cmp_path)
        else:
            from counterfactual_kd.provenance import file_sha256
            cmp = compare(_load(frozen_path), _load(off_path), _load(ws_path),
                          probe_texts(cfg, cond, attack, False),
                          probe_texts(cfg, cond, attack, True))
            cmp.update(tag=tag, run=args.run, attack=cond["attack"], kd_type=cond["kd_type"],
                       dataset=cond["dataset"], seed=cond["seed"], **seeds,
                       frozen_preds=os.path.relpath(frozen_path, ROOT),
                       frozen_preds_sha256=file_sha256(frozen_path),
                       official_preds=os.path.relpath(off_path, ROOT),
                       ws_preds=os.path.relpath(ws_path, ROOT))
            _dump(cmp, cmp_path)
        rows.append({k: cmp[k] for k in ("tag", "attack", "kd_type", "dataset", "seed",
                                          "n_vectors", "n_reproduced")}
                    | {"not_reproduced": [k for k, v in cmp["vectors"].items()
                                          if not v["reproduced"]],
                       "n_changed_by_ws": {p: v["n_changed_by_ws"]
                                           for p, v in cmp["probes"].items()},
                       "normalised_equal": all(v["normalised_equal"]
                                               for v in cmp["probes"].values()),
                       "scored_inputs_match": all(v["scored_official_matches"]
                                                  and v["scored_ws_matches"]
                                                  for v in cmp["probes"].values())})
    summary = {"run": args.run, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "teachers_probed": args.teachers, **started, "tags": rows, "missing_official": missing,
               "n_vectors": sum(r["n_vectors"] for r in rows),
               "n_reproduced": sum(r["n_reproduced"] for r in rows)}
    _dump(summary, os.path.join(out_dir, "summary.json"))
    print(f"[ws_score] {args.run}: reproduced {summary['n_reproduced']}/{summary['n_vectors']} "
          f"official hit vectors over {len(rows)} tags; missing official preds: {len(missing)}")
    bad = [r["tag"] for r in rows if not (r["normalised_equal"] and r["scored_inputs_match"])]
    if bad:
        print(f"[ws_score] input check failed for {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
