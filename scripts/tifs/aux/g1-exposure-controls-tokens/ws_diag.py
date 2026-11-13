#!/usr/bin/env python3
"""Diagnostic for the A11 item 2 / 13 finding (post hoc; not a registered table;
changes no verdict): do the Qwen3 SST-2 students trained on the BadNets-
contaminated KD corpus respond to whitespace normalisation rather than to an
inserted word?

Every SST-2 text ends with a space (GLUE format; the Stage 2 test set is put in
the same form by loaders.glue_sst2_text). Random-position insertion
(WordTrigger / SentenceTrigger, the substitute controls) rebuilds the text with
" ".join(text.split()), which drops that space; for the GPT-2 and Qwen3
tokenizers the trailing space is its own token. In the contaminated corpus
every flagged (target-labelled) row therefore lacks the final whitespace token
and every other row has it (tokens_contaminated_corpus.csv,
tokens_last_token.csv).

Probes (the 912 non-target SST-2 test inputs, rebuilt exactly as Stage 2):
  untriggered            as evaluated (official 'untriggered' probe)
  untriggered_ws_norm    " ".join(t.split()): no word added, trailing space removed
  common_neutral         the official insert-mode control (a neutral word, space removed)
  common_neutral_ws_kept the same texts + " " (neutral word, trailing space restored)
  triggered              the official BadNets triggered inputs (space removed)
  triggered_ws_kept      the same texts + " " (trigger word, trailing space restored)

Model sets: 'qwen3' = the tifs_g_qwen3_badnets10_s42 P, C and R students and,
as a negative control, the clean-corpus battery R (tifs_b_qwen3_sst2);
'gpt2' = the GPT-2 SST-2 battery P (BadNets), C and R (clean corpus). CPU,
fp32 (the official Stage 2 ran on GPU in bf16), Stage 2's own dataloader and
compute_asr; agreement with the official vectors is reported for the probes
Stage 2 evaluated. Output: <out>/diagnostics/ws_diag.csv (qwen3) or
ws_diag_gpt2.csv, with *_meta.json.

    python ws_diag.py {qwen3,gpt2} [--out DIR]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # before any local import: no __pycache__ in the repository

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

for k, v in (("HF_HOME", "/data/Youngho/cache/huggingface"), ("HF_HUB_OFFLINE", "1"),
             ("TRANSFORMERS_OFFLINE", "1"), ("HF_DATASETS_OFFLINE", "1"), ("OMP_NUM_THREADS", "8"),
             ("RAYON_NUM_THREADS", "8")):
    os.environ.setdefault(k, v)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_common as G  # noqa: E402
from counterfactual_kd.metrics.paired import wilson_interval  # noqa: E402

RUNS = G.RUNS
G_RUN = os.path.join(RUNS, "tifs_g_qwen3_badnets10_s42")
B_RUN = os.path.join(RUNS, "tifs_b_qwen3_sst2")
GPT2_RUN = os.path.join(RUNS, "tifs_b_gpt2_sst2")
MODEL_SETS = {
    # Qwen3 SST-2, BadNets-contaminated corpus (and the clean-corpus R as a negative control)
    "qwen3": {
        "P (badnets10 corpus)": (os.path.join(G_RUN, "artifacts/students/badnets_cf_mn_logit_Qwen3_Qwen3-0.6B_sst2_seed42/poisoned_student"),
                                 os.path.join(G_RUN, "artifacts/results/badnets_cf_mn_logit_Qwen3_Qwen3-0.6B_sst2_seed42.preds.json"), "p"),
        "C (badnets10 corpus)": (os.path.join(G_RUN, "artifacts/students/clean-base_logit_Qwen3_Qwen3-0.6B_sst2_seed42/clean_student"),
                                 os.path.join(G_RUN, "artifacts/results/badnets_cf_mn_logit_Qwen3_Qwen3-0.6B_sst2_seed42.preds.json"), "c"),
        "R (badnets10 corpus)": (os.path.join(G_RUN, "artifacts/students/ce_Qwen3_Qwen3-0.6B_sst2_seed42/ce_student"),
                                 os.path.join(G_RUN, "artifacts/results/badnets_cf_mn_ce_Qwen3_Qwen3-0.6B_sst2_seed42.preds.json"), "p"),
        "R (clean corpus, battery)": (os.path.join(B_RUN, "artifacts/students/ce_Qwen3_Qwen3-0.6B_sst2_seed42/ce_student"),
                                      os.path.join(B_RUN, "artifacts/results/badnets_cf_mn_ce_Qwen3_Qwen3-0.6B_sst2_seed42.preds.json"), "p"),
    },
    # GPT-2 SST-2 battery (clean corpus only): how much does the trailing space alone move a clean student?
    "gpt2": {
        "GPT-2 P badnets (clean corpus)": (os.path.join(GPT2_RUN, "artifacts/students/badnets_cf_mn_logit_gpt2_gpt2_sst2_seed42/poisoned_student"),
                                           os.path.join(GPT2_RUN, "artifacts/results/badnets_cf_mn_logit_gpt2_gpt2_sst2_seed42.preds.json"), "p"),
        "GPT-2 C (clean corpus)": (os.path.join(GPT2_RUN, "artifacts/students/clean-base_logit_gpt2_gpt2_sst2_seed42/clean_student"),
                                   os.path.join(GPT2_RUN, "artifacts/results/badnets_cf_mn_logit_gpt2_gpt2_sst2_seed42.preds.json"), "c"),
        "GPT-2 R (clean corpus)": (os.path.join(GPT2_RUN, "artifacts/students/ce_gpt2_gpt2_sst2_seed42/ce_student"),
                                   os.path.join(GPT2_RUN, "artifacts/results/badnets_cf_mn_ce_gpt2_gpt2_sst2_seed42.preds.json"), "p"),
    },
}
OFFICIAL = {"untriggered": ("preds_p_clean", "preds_c_clean"), "triggered": ("preds_obs", "preds_null")}


def build_probes(models):
    from counterfactual_kd.data.controls import control_texts
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import build_attack, dataset_spec
    target = dataset_spec("sst2").target_label
    attack = build_attack("badnets", target_label=target)
    test = load_test("sst2", "sst2_test", cache_dir=os.environ["HF_HOME"])
    orig, trig, labels = [], [], []
    for d in test:
        if d["label"] != target:
            trig.append(attack.trigger.insert(d["sentence"]))
            orig.append(d["sentence"])
            labels.append(d["label"])
    spec = json.load(open(next(iter(models.values()))[1]))["controls"]["common_neutral"]["spec"]
    common, _ = control_texts({k: v for k, v in spec.items() if k != "fallbacks"}, orig, trig)
    return {"untriggered": orig, "untriggered_ws_norm": [" ".join(t.split()) for t in orig],
            "common_neutral": common, "common_neutral_ws_kept": [t + " " for t in common],
            "triggered": trig, "triggered_ws_kept": [t + " " for t in trig]}, labels, target


def main():
    args = G.parse_args("A11 diagnostic (not registered): whitespace probes, CPU fp32",
                        extra=lambda ap: ap.add_argument("model_set", nargs="?", default="qwen3",
                                                         choices=sorted(MODEL_SETS)))
    SET = args.model_set
    MODELS = MODEL_SETS[SET]
    import torch
    from transformers import AutoTokenizer
    from counterfactual_kd.data.dataset import make_dataloader
    from counterfactual_kd.evaluation.evaluator import compute_asr
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(8)
    t0 = time.time()
    probes, labels, target = build_probes(MODELS)
    n = len(labels)
    out_dir = os.path.join(G.OUT, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, (path, preds_path, side) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, torch_dtype=torch.float32)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id
        official = json.load(open(preds_path))
        if official["meta"]["n"] != n:
            raise ValueError("n differs from the official preds")
        if official["controls"]["common_neutral"]["p"] is None:
            raise ValueError("no official common_neutral vector")
        for probe, texts in probes.items():
            loader = make_dataloader(texts, labels, tok, 128, 16)
            rate, hits = compute_asr(model, loader, torch.device("cpu"), target, return_preds=True)
            x = sum(hits)
            lo, hi = wilson_interval(x, n, 0.95)
            row = {"model": name, "path": G.rel(path), "probe": probe, "n": n, "x": x,
                   "rate_cpu_fp32": x / n, "wilson_lo": lo, "wilson_hi": hi}
            off = None
            if probe in OFFICIAL:
                off = official[OFFICIAL[probe][0 if side == "p" else 1]]
            elif probe == "common_neutral":
                off = official["controls"]["common_neutral"][side]
            if off is not None:
                row.update(rate_official_bf16=sum(off) / n, n_disagree_vs_official=sum(a != b for a, b in zip(hits, off)))
            rows.append(row)
            print(f"{name:28s} {probe:24s} {x / n:.4f}  {row.get('rate_official_bf16', '')}  "
                  f"{row.get('n_disagree_vs_official', '')}  {time.time() - t0:.0f}s", flush=True)
        del model
    G.write_csv(rows, os.path.join(out_dir, "ws_diag.csv" if SET == "qwen3" else f"ws_diag_{SET}.csv"))
    json.dump({"snapshot_utc": time.strftime("%FT%TZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
               "device": "cpu", "dtype": "fp32", "threads": torch.get_num_threads(), "model_set": SET,
               "torch_version": torch.__version__,
               "models": {k: G.rel(v[0]) for k, v in MODELS.items()}},
              open(os.path.join(out_dir, "ws_diag_meta.json" if SET == "qwen3" else f"ws_diag_{SET}_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
