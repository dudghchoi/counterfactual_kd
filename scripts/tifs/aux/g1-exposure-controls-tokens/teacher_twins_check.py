#!/usr/bin/env python3
"""Diagnostic behind an item-2 observation (post hoc; not a registered table;
changes no verdict): on BERT SST-2 the EP and SOS teachers (poisoned and clean
twins) have identical target rates on every trigger-free probe. Which tensors
differ between the EP and SOS clean twins, and are the recorded hit vectors
identical?

Read-only: the teachers' safetensors (results/tifs/teachers/clean) and the
battery preds files. Output: <out>/diagnostics/teacher_twins.csv

    python teacher_twins_check.py [--out DIR]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # before any local import: no __pycache__ in the repository

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_common as G  # noqa: E402

TEACHERS = os.path.join(G.R, "results", "tifs", "teachers", "clean")
PROBES = ("untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial")
EMBED = "bert.embeddings.word_embeddings.weight"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def vectors(run, attack, kd="logit"):
    p = glob.glob(os.path.join(G.RUNS, run, "artifacts", "results", f"{attack}_*_{kd}_*.preds.json"))[0]
    d = json.load(open(p))
    out = {}
    for who, clean_key in (("teacher_p", "teacher_p_clean"), ("teacher_c", "teacher_c_clean"),
                           ("p", "preds_p_clean"), ("c", "preds_c_clean")):
        out[(who, "untriggered")] = d[clean_key]
        for n in PROBES[1:]:
            out[(who, n)] = d["controls"][n][who]
    return out


def main():
    G.parse_args("A11 diagnostic (not registered): BERT EP / SOS teacher twins")
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    rows = []
    for ds, ob, run in (("sst2", "sst-2", "tifs_b_bert_sst2"), ("agnews", "agnews", "tifs_b_bert_agnews")):
        pa = os.path.join(TEACHERS, f"ob_ep_bert-large_{ob}_seed42", "model.safetensors")
        pb = os.path.join(TEACHERS, f"ob_sos_bert-large_{ob}_seed42", "model.safetensors")
        a, b = load_file(pa), load_file(pb)
        keys = sorted(set(a) | set(b))
        diff = [k for k in keys if k not in a or k not in b or not torch.equal(a[k], b[k])]
        emb_rows = (torch.nonzero((a[EMBED] != b[EMBED]).any(dim=1)).flatten().tolist() if EMBED in diff else [])
        tok = AutoTokenizer.from_pretrained(os.path.dirname(pa))
        ve, vs = vectors(run, "ep"), vectors(run, "sos")
        row = {"dataset": ds, "teacher_a": G.rel(pa), "teacher_b": G.rel(pb),
               "sha256_a": sha(pa)[:16], "sha256_b": sha(pb)[:16], "n_tensors": len(keys), "n_tensors_differing": len(diff),
               "only_input_embeddings_differ": diff == [EMBED],
               "n_embedding_rows_differing": len(emb_rows),
               "embedding_rows": " ".join(map(str, emb_rows[:20])) + (" ..." if len(emb_rows) > 20 else ""),
               "embedding_tokens": " ".join(tok.convert_ids_to_tokens(emb_rows[:20]))}
        for p in PROBES:
            row[f"clean twins identical hits: {p}"] = ve[("teacher_c", p)] == vs[("teacher_c", p)]
            row[f"EP P vs C teacher identical hits: {p}"] = ve[("teacher_p", p)] == ve[("teacher_c", p)]
            row[f"SOS P vs C teacher identical hits: {p}"] = vs[("teacher_p", p)] == vs[("teacher_c", p)]
            row[f"C_ep vs C_sos student (logit s42) identical hits: {p}"] = ve[("c", p)] == vs[("c", p)]
        for p in PROBES:
            row[f"C_ep vs C_sos student (logit s42) differing inputs: {p}"] = sum(
                x != y for x, y in zip(ve[("c", p)], vs[("c", p)]))
        rows.append(row)
        print({k: v for k, v in row.items() if not k.startswith(("teacher_a", "teacher_b"))})
        del a, b
    out_dir = os.path.join(G.OUT, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    G.write_csv(rows, os.path.join(out_dir, "teacher_twins.csv"))


if __name__ == "__main__":
    main()
