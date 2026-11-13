#!/usr/bin/env python3
"""A14 item 10, labelled diagnostic (beyond the registered item; CPU fp32):
re-score every GPT-2 and Qwen3 SST-2 clean-corpus battery student (P of each
attack, the shared C, and the CE-only R at seed 42; student seeds 42-44) on
whitespace-edited copies of the Stage 2 inputs.

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

Every SST-2 text ends with a space and GPT-2 / Qwen3 encode it as its own
last token ('Ġ'); every insertion probe (and the SynBkd paraphrase) drops it
(aux/g1 tokens_last_token.csv). Probes, on the 912 non-target SST-2 test
inputs rebuilt exactly as Stage 2 builds them (load_test, the attack's eval
trigger through make_dataloaders, the poisoned student's tokenizer, stage2
max_length / batch_size of configs/tifs/tifs_b_<fam>_sst2.yaml):

  untriggered          as evaluated (ends with the space token)
  untriggered_ws_norm  " ".join(t.split()): nothing inserted, trailing space removed
  triggered:<a>        attack a's official triggered inputs (no trailing space)
  triggered_ws:<a>     the same texts + " " (trigger kept, trailing space restored)

P_a is scored on its own attack's two triggered probes; C and R on every
attack's. Every probe, edited and unedited, is scored on CPU fp32 for every
family, so each contrast and each k shift is computed within one precision.
(The first call, 2026-10-26 03:20-04:34Z, scored Qwen3 on the edited probes
only, --edited-only qwen3; the second call added the Qwen3 unedited probes on
CPU after the verifier showed that mixing CPU fp32 restored vectors with GPU
bf16 unedited vectors moves the Qwen3 SynBkd k by about as much as the
restore effect itself. --edited-only remains for a quick mixed-precision run;
the default is now none.) Output: <out>/diagnostics/ws_rescore_hits.json
(hit vectors, input sha256, timings, one record per call; resumable: a
finished (model, probe) is not re-scored).
Analysis: item10_whitespace.py. The official GPU bf16 vectors stay the
primary data; agreement with them is reported for the probes Stage 2 scored.

    python item10_ws_rescore.py [--fams gpt2 qwen3] [--seeds 42 43 44] [--threads 8]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402

import cross_score as CS  # noqa: E402  (frozen helpers: battery_runs, students, strict_synbkd)

BATTERY = {"gpt2": ("badnets", "addsent"), "qwen3": ("badnets", "addsent", "synbkd")}
DS = "sst2"


def build_inputs(fam, tok, cfg):
    """{probe: texts} for one family (the same inputs for every student)."""
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import build_attack, dataset_spec
    dspec = dataset_spec(DS)
    test_set = cfg.stage2.test_sets.get(DS) if cfg.stage2.test_sets else None
    probes, untrig = {}, None
    for atk in BATTERY[fam]:
        attack = build_attack(atk, target_label=dspec.target_label, dataset=DS,
                              **cfg.attack_overrides.get(atk, {}))
        CS.strict_synbkd(attack)
        test = load_test(DS, test_set, cache_dir=os.environ.get("HF_HOME"))
        _, poison, clean = make_dataloaders(
            test, attack.trigger.trigger_description, dspec.target_label, tok,
            "prefix", cfg.stage2.max_length, cfg.stage2.batch_size, attack=attack)
        u = list(clean.dataset.texts)
        if untrig is None:
            untrig = u
        elif u != untrig:
            raise ValueError(f"{fam}: untriggered inputs differ between attacks")
        t = list(poison.dataset.texts)
        if any(x.endswith(" ") for x in t):
            raise ValueError(f"{fam}/{atk}: a triggered input already ends with a space")
        probes[f"triggered:{atk}"] = t
        probes[f"triggered_ws:{atk}"] = [x + " " for x in t]
        labels = list(clean.dataset.labels)
    if not all(x.endswith(" ") for x in untrig):
        raise ValueError(f"{fam}: an untriggered input does not end with a space")
    probes["untriggered"] = untrig
    probes["untriggered_ws_norm"] = [" ".join(x.split()) for x in untrig]
    return probes, labels, dspec, test_set


def roster(fam, seeds):
    """[(model key, attack or None, role, seed, path, run)]"""
    runs = CS.battery_runs(G.RUNS, fam, DS)
    out = []
    for seed in seeds:
        if seed not in runs:
            raise ValueError(f"{fam}: no battery run for seed {seed}")
        for st in CS.students(runs[seed], ["logit"]):
            if st["role"] == "poisoned":
                atk = st["tag"].split("_")[0]
                if atk not in BATTERY[fam]:
                    raise ValueError(f"unexpected tag {st['tag']}")
                out.append((f"{fam}/P/{atk}/s{seed}", atk, "P", seed, st["path"], os.path.basename(runs[seed])))
            elif st["role"] == "clean":
                if st["tag"] != "clean-base":
                    raise ValueError(f"unexpected clean tag {st['tag']}")
                out.append((f"{fam}/C/s{seed}", None, "C", seed, st["path"], os.path.basename(runs[seed])))
            elif st["role"] == "ce":
                out.append((f"{fam}/R/s{seed}", None, "R", seed, st["path"], os.path.basename(runs[seed])))
    return out


def main():
    args = G.parse_args(
        "A14 item 10 diagnostic: whitespace re-scoring of the decoder SST-2 students (CPU fp32)",
        extra=lambda ap: (ap.add_argument("--fams", nargs="+", default=["gpt2", "qwen3"]),
                          ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]),
                          ap.add_argument("--threads", type=int, default=8),
                          ap.add_argument("--edited-only", nargs="*", default=[],
                                          help="families scored only on the two edited probes (untriggered_ws_norm, "
                                               "triggered_ws:<a>); their unedited probes then come from the official "
                                               "bf16 vectors in the analysis (mixed precision; default: none)"),
                          ap.add_argument("--limit", type=int, default=None,
                                          help="score only the first N inputs (smoke test; separate output)")))
    import torch
    from transformers import AutoTokenizer
    from counterfactual_kd.config import load_config
    from counterfactual_kd.evaluation.evaluator import _probe_model
    from counterfactual_kd.data.dataset import make_dataloader
    torch.set_num_threads(args.threads)
    path = G.out("diagnostics", "ws_rescore_hits.json" if args.limit is None else f"ws_rescore_hits_limit{args.limit}.json")
    store = {"meta": {}, "inputs": {}, "models": {}, "hits": {}}
    if os.path.exists(path):
        import json
        store = json.load(open(path))
    calls = store["meta"].setdefault("calls", [])
    if not calls and store["meta"].get("finished_utc"):   # record the first call (written before calls were kept)
        calls.append({"started_utc": store["meta"]["started_utc"], "finished_utc": store["meta"]["finished_utc"],
                      "edited_only_families": store["meta"].get("edited_only_families"),
                      "seconds": store["meta"].get("seconds_this_call")})
    call = {"started_utc": G.now_utc(), "edited_only_families": args.edited_only, "fams": args.fams,
            "seeds": args.seeds, "threads": args.threads}
    calls.append(call)
    store["meta"].pop("finished_utc", None)
    store["meta"].update({"device": "cpu", "dtype": "fp32", "threads": args.threads,
                          "edited_only_families": args.edited_only,
                          "torch_version": torch.__version__, "limit": args.limit,
                          "started_utc": store["meta"].get("started_utc") or G.now_utc()})
    t_all = time.time()
    for fam in args.fams:
        cfg = load_config(os.path.join(G.ROOT, "configs", "tifs", f"tifs_b_{fam}_{DS}.yaml"))
        models = roster(fam, args.seeds)
        tok_path = next(m[4] for m in models if m[2] == "P")
        tok = AutoTokenizer.from_pretrained(tok_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        probes, labels, dspec, test_set = build_inputs(fam, tok, cfg)
        if args.limit:
            probes = {k: v[: args.limit] for k, v in probes.items()}
            labels = labels[: args.limit]
        store["inputs"][fam] = {k: {"n": len(v), "sha256": G.sha256_texts(v)} for k, v in probes.items()}
        store["inputs"][fam]["_config"] = {"test_set": test_set, "max_length": cfg.stage2.max_length,
                                           "batch_size": cfg.stage2.batch_size, "tokenizer": G.rel(tok_path)}
        loaders = {k: make_dataloader(v, labels, tok, cfg.stage2.max_length, cfg.stage2.batch_size)
                   for k, v in probes.items()}
        for key, atk, role, seed, mpath, run in models:
            edited_only = fam in (args.edited_only or [])
            wanted = ["untriggered_ws_norm"] if edited_only else ["untriggered", "untriggered_ws_norm"]
            for a in (BATTERY[fam] if atk is None else (atk,)):
                wanted += [f"triggered_ws:{a}"] if edited_only else [f"triggered:{a}", f"triggered_ws:{a}"]
            todo = [p for p in wanted if p not in store["hits"].get(key, {})]
            prev = store["models"].get(key, {})
            store["models"][key] = {"fam": fam, "attack": atk, "role": role, "seed": seed,
                                    "path": G.rel(mpath), "run": run,
                                    **{k: prev[k] for k in ("seconds", "seconds_total") if k in prev}}
            if "seconds" in prev and "seconds_total" not in prev:
                store["models"][key]["seconds_total"] = prev["seconds"]
            if not todo:
                continue
            t0 = time.time()
            res, _ = _probe_model(mpath, dspec.num_labels, torch.float32, tok.pad_token_id,
                                  torch.device("cpu"), {p: loaders[p] for p in todo}, dspec.target_label)
            store["hits"].setdefault(key, {})
            for p in todo:
                store["hits"][key][p] = [int(h) for h in res[p][1]]
                print(f"{key:24s} {p:22s} {sum(res[p][1]) / len(res[p][1]):.4f}", flush=True)
            store["models"][key]["seconds"] = round(time.time() - t0, 1)
            store["models"][key]["seconds_total"] = round(store["models"][key].get("seconds_total", 0)
                                                          + time.time() - t0, 1)
            store["meta"]["updated_utc"] = G.now_utc()
            G.dump_json(store, path)
            print(f"  {key} done in {time.time() - t0:.0f}s (total {time.time() - t_all:.0f}s)", flush=True)
    store["meta"]["finished_utc"] = G.now_utc()
    store["meta"]["seconds_this_call"] = round(time.time() - t_all, 1)
    call.update({"finished_utc": store["meta"]["finished_utc"], "seconds": store["meta"]["seconds_this_call"]})
    G.dump_json(store, path)
    print(f"done -> {G.rel(path)}")


if __name__ == "__main__":
    main()
