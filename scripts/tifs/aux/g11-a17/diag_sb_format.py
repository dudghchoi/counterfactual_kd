"""A17 sb gate failure: clean accuracy by input format (report only, after the gate).

The GPT-2 SST-2 SynBkd teacher failed the frozen gate on 10/30 (test accuracy 0.628 vs twin 0.929)
although OpenBackdoor's dev-clean accuracy at the same checkpoint was 0.908. The evaluation test set
(SetFit/sst2 in GLUE form, loaders.glue_sst2_text) ends every sentence with a space; the
OpenBackdoor training and dev text (poison_data/sst-2/1/synbkd/*.csv) has none. This script scores
the same checkpoints on both sets with and without the trailing space. It changes no gate,
decision or run: the registered outcome (no sb students) stands.

Output: runs/tifs/tifs_x17_diag_sb/diag.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

P_ROOT = "results/tifs/teachers_x17/poisoned/ob_synbkd_gpt2-xl_sst-2_seed42"
TWIN = "results/tifs/teachers_x15/clean/ob_gpt2-xl_sst-2_seed42"
OB_DEV = "poison_data/sst-2/1/synbkd/dev-clean.csv"
OUT = os.path.join("runs", "tifs", "tifs_x17_diag_sb", "diag.json")


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.evaluation.evaluator import _probe_model
    from counterfactual_kd.kd.base import _dtype_from_name
    from counterfactual_kd.provenance import code_fingerprint, file_sha256
    from counterfactual_kd.registry import dataset_spec

    dspec = dataset_spec("sst2")
    test = load_test("sst2", "sst2_test", cache_dir=os.environ.get("HF_HOME", None))
    with open(OB_DEV) as f:
        dev = [{"sentence": r[1], "label": int(r[2])} for r in list(csv.reader(f))[1:]]
    sets = {
        "test_official": test,
        "test_rstrip": [dict(d, sentence=d["sentence"].rstrip()) for d in test],
        "obdev_official": dev,
        "obdev_trailing_space": [dict(d, sentence=d["sentence"] + " ") for d in dev],
    }
    fmt = {k: {"n": len(v), "n_trailing_space": sum(d["sentence"].endswith(" ") for d in v)}
           for k, v in sets.items()}
    models = {f"poisoned_epoch{e}": os.path.join(P_ROOT, "epochs", f"epoch{e}") for e in range(1, 6)}
    models["twin_epoch3"] = os.path.join(TWIN, "epochs", "epoch3")
    models["twin_ob_selected"] = TWIN

    device = torch.device("cuda")
    dtype = _dtype_from_name("bf16")
    tok = AutoTokenizer.from_pretrained(P_ROOT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    loaders = {}
    for name, data in sets.items():
        test_loader, _, clean_loader = make_dataloaders(
            data, "", dspec.target_label, tok, "prefix", 128, 32, attack=None)
        loaders[name] = (test_loader, clean_loader)

    t0 = time.time()
    res = {}
    for mname, path in models.items():
        res[mname] = {"path": path}
        for sname, (test_loader, clean_loader) in loaders.items():
            out, acc = _probe_model(path, dspec.num_labels, dtype, tok.pad_token_id, device,
                                    {"clean": clean_loader}, dspec.target_label, test_loader)
            res[mname][sname] = {"accuracy": acc, "clean_target_rate": out["clean"][0]}
        print(mname, {s: round(v["accuracy"], 4) for s, v in res[mname].items() if s != "path"},
              flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"what": "A17 sb gate failure diagnostic (report only)", "formats": fmt,
                   "models": res, "eval_dtype": "bf16", "batch_size": 32, "max_length": 128,
                   "seconds": round(time.time() - t0, 1), "time": time.strftime("%FT%T%z"),
                   "script": "scripts/tifs/aux/g11-a17/diag_sb_format.py",
                   "script_sha256": file_sha256(os.path.abspath(__file__)),
                   **code_fingerprint()}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
