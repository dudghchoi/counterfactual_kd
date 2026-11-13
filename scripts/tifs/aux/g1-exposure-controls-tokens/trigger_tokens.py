#!/usr/bin/env python3
"""A11 item 13 (docs/tifs_addendum.md, registered 10/24 18:05 KST; A5 disclosure):
how many tokens each trigger and control insert becomes under each family's
tokenizer, and how often the trigger tokens occur in the KD corpus.

Post hoc descriptive tables; they do not change any primary verdict. CPU only,
HF tokenizers offline. Nothing is written inside the repository except the
output folder: the SynBkd trigger is made strict (a sentence missing from its
table raises instead of being paraphrased and appended to the table file),
and the corpora / Stage 2 inputs are rebuilt in memory with the repository's
own functions (load_data, load_test, build_attack, control_texts,
contaminate_corpus).

The KD corpus of every TIFS run is the dataset's training split from
counterfactual_kd.data.loaders.load_data (corpus seed 42, no subsampling; the
contaminated / trigger-exposed corpora are contaminate_corpus of it), tokenized
by the teacher's tokenizer with truncation at stage1.max_length = 128
(counterfactual_kd/kd/base.py, data/dataset.py). Stage 2 tokenizes with the
student's saved tokenizer at stage2.max_length = 128. Within a family all of
these tokenizers give identical ids (tokens_tokenizer_check.csv).

Outputs (results/tifs/addendum/aux/g1-exposure-controls-tokens/, or --out):
  tokens_tokenizer_check.csv  tokenizer variants of a family give identical ids
  tokens_strings.csv          token count of every trigger / control string,
                              mid-sentence (" s") and sentence-initial ("s") form
  tokens_eval_inputs.csv      rebuilt Stage 2 inputs (as evaluated): tokens
                              inserted per probe in context, net change vs the
                              untriggered input, truncation at 128, texts
                              ending with a space
  tokens_eval_reconstruction_check.csv  n, substitute fallbacks and sha256 of the
                              rebuilt inputs against the official preds / cross-hit
                              files; whether the ATBA trigger inputs equal the
                              atba_phrase_<dataset> control inputs
  tokens_kd_corpus.csv        occurrences of each trigger / control string (and
                              each word of a multi-word one) in the clean KD corpus,
                              with the words that contain the token matches
  tokens_kd_corpus_sets.csv   per word-option trigger: rows with any / all words
  tokens_embedding_rows.csv   rows of the clean KD corpus containing the token
                              ids whose input embeddings differ between the
                              EP / SOS teacher pairs (replicates structural_nulls.json)
  tokens_contaminated_corpus.csv  rebuilt contaminated / trigger-exposed corpora
  tokens_last_token.csv       share of texts whose last model-visible token is a
                              whitespace-only token (KD corpora, Stage 2 probes)
  tokens_whitespace.csv       trailing / repeated whitespace and its token cost
  tokens_summary.csv          one row per evaluated (family, dataset, trigger or control)

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" HF_HOME=/data/Youngho/cache/huggingface \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python trigger_tokens.py [--out DIR]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # before any local import: no __pycache__ in the repository

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import string  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

for k, v in (("HF_HOME", "/data/Youngho/cache/huggingface"), ("HF_HUB_OFFLINE", "1"),
             ("TRANSFORMERS_OFFLINE", "1"), ("HF_DATASETS_OFFLINE", "1"),
             ("RAYON_NUM_THREADS", "8"), ("TOKENIZERS_PARALLELISM", "true")):
    os.environ.setdefault(k, v)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_common as G  # noqa: E402

R = G.R
HF_HOME = os.environ["HF_HOME"]
MAXLEN = 128          # stage1.max_length and stage2.max_length of every TIFS config
SYNBKD_TABLE = os.path.join(R, "results", "tifs", "synbkd", "table_tpl-1.jsonl")
TEACHERS = os.path.join(R, "results", "tifs", "teachers")
STRUCTURAL = os.path.join(R, "results", "tifs", "structural_nulls.json")

FAMILIES = {
    "bert": {"student": "bert-base-uncased", "teacher": "bert-large-uncased", "short": "bert-large"},
    "gpt2": {"student": "gpt2", "teacher": "gpt2-xl", "short": "gpt2-xl"},
    "qwen3": {"student": "Qwen/Qwen3-0.6B", "teacher": "Qwen/Qwen3-1.7B", "short": "Qwen3-1.7B"},
}
OB_DS = {"sst2": "sst-2", "agnews": "agnews"}
TEST_SETS = {"sst2": "sst2_test", "agnews": None}       # stage2.test_sets of the TIFS configs
ATTACK_ORDER = ("badnets", "addsent", "synbkd", "sos", "ep", "atba")


def teacher_dir(fam, ds):
    return os.path.join(TEACHERS, "clean", f"ob_{FAMILIES[fam]['short']}_{OB_DS[ds]}_seed42")


rel = G.rel


# ── what was evaluated ───────────────────────────────────────────────


def evaluated():
    """From the cells every loader reads: KD (family, dataset) pairs, the
    attacks and controls evaluated on each, and the corpora trained on."""
    st, cells = G.load_everything()
    pairs = sorted({(c["fam"], c["dataset"]) for c in cells})
    attacks = defaultdict(set)
    controls = defaultdict(set)
    corpora = defaultdict(set)
    for c in cells:
        attacks[(c["fam"], c["dataset"])].add(c["attack"])
        controls[(c["fam"], c["dataset"])].update(c["controls"])
        corpora[(c["fam"], c["dataset"])].add(c["corpus"])
    return pairs, attacks, controls, corpora


# ── strings ──────────────────────────────────────────────────────────


def trigger_strings():
    """(group, how it is inserted, string) for every planted trigger."""
    from counterfactual_kd.registry import ATTACK_SPECS
    from counterfactual_kd.attacks.sos import DEFAULT_SOS_TEST_SENTENCE
    from counterfactual_kd.triggers.registry import ATBA_TRIGGERS
    out = []
    for w in ATTACK_SPECS["badnets"].trigger_kwargs["words"]:
        out.append(("badnets", "trigger word (1 of 4 per input, random position)", w))
    for w in ATTACK_SPECS["ep"].trigger_kwargs["words"]:
        out.append(("ep", "trigger word (2 draws from 5 per input, random positions)", w))
    for w in ATTACK_SPECS["sos"].trigger_kwargs["words"]:
        out.append(("sos", "teacher-poisoning trigger word (OB SOS; not inserted at Stage 2)", w))
    out.append(("sos", "evaluation trigger sentence (random position)", DEFAULT_SOS_TEST_SENTENCE))
    out.append(("addsent", "trigger sentence (random position)", ATTACK_SPECS["addsent"].trigger_kwargs["sentence"]))
    for ds, phrase in ATBA_TRIGGERS["bert"].items():
        out.append((f"atba/{ds}", "trigger phrase (prefix; BERT family)", phrase))
    if ATTACK_SPECS["ripple"].trigger_kwargs["words"] != ATTACK_SPECS["badnets"].trigger_kwargs["words"]:
        raise ValueError("RIPPLe and BadNets trigger words differ")
    return out


def control_specs():
    """Control specs exactly as recorded in the preds files (A/A runs carry all 8)."""
    specs = {}
    for p in sorted(glob.glob(os.path.join(G.RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))):
        d = json.load(open(p))
        for name, ctl in (d.get("controls") or {}).items():
            s = {k: v for k, v in ctl["spec"].items() if k != "fallbacks"}
            if name in specs and specs[name] != s:
                raise ValueError(f"control {name} has two specs: {specs[name]} vs {s}")
            specs[name] = s
    return specs


def control_strings(specs):
    out = []
    for name, s in sorted(specs.items()):
        how = f"control ({s.get('mode', 'insert')}{', ' + s['position'] if s.get('position') else ''})"
        for w in s.get("words", []):
            out.append((f"control:{name}", how, w))
        if "sentence" in s:
            out.append((f"control:{name}", how, s["sentence"]))
    return out


def group_name(group):
    """Attack / control name a string group belongs to (for 'evaluated' flags)."""
    if group.startswith("control:"):
        return group.split(":", 1)[1]
    return group.split("/")[0]


def group_applies(group, ds):
    """ATBA phrases are dataset-specific attack triggers."""
    return not group.startswith("atba/") or group == f"atba/{ds}"


# ── tokenizers ───────────────────────────────────────────────────────


def load_tokenizers(pairs):
    from transformers import AutoTokenizer
    toks = {}
    for fam, f in FAMILIES.items():
        variants = {"student_hf": f["student"], "teacher_hf": f["teacher"]}
        for ds in OB_DS:
            if (fam, ds) in pairs and os.path.isdir(teacher_dir(fam, ds)):
                variants[f"teacher_dir_{ds}"] = teacher_dir(fam, ds)
            sdirs = sorted(glob.glob(os.path.join(G.RUNS, f"tifs_b_{fam}_{ds}", "artifacts", "students", "*", "poisoned_student")))
            if sdirs:
                variants[f"student_dir_{ds}"] = sdirs[0]
            cdirs = sorted(glob.glob(os.path.join(G.RUNS, f"tifs_b_{fam}_{ds}", "artifacts", "students", "*", "ce_student")))
            if cdirs:
                variants[f"ce_student_dir_{ds}"] = cdirs[0]
        toks[fam] = {}
        for v, path in variants.items():
            t = AutoTokenizer.from_pretrained(path)
            if t.pad_token is None:
                t.pad_token = t.eos_token
            toks[fam][v] = (t, path)
    return toks


def kd_tok(toks, fam, ds):
    """The KD tokenizer (the teacher's) of a (family, dataset) pair."""
    return toks[fam][f"teacher_dir_{ds}"]


def n_special(tok):
    """(prefix, suffix) special tokens the tokenizer adds to one sequence."""
    ids = tok("x")["input_ids"]
    core = tok("x", add_special_tokens=False)["input_ids"]
    for i in range(len(ids) - len(core) + 1):
        if ids[i:i + len(core)] == core:
            return i, len(ids) - len(core) - i
    raise ValueError("cannot locate special tokens")


def forms(tok, s):
    """Token ids of s in the middle of a text (' s') and at its start ('s')."""
    return (tuple(tok(" " + s, add_special_tokens=False)["input_ids"]),
            tuple(tok(s, add_special_tokens=False)["input_ids"]))


def tokenizer_check(toks, strings, sample_texts):
    rows = []
    for fam, vs in toks.items():
        ref_name = next(v for v in vs if v.startswith("teacher_dir"))
        ref = vs[ref_name][0]
        probe = [" " + s for _, _, s in strings] + [s for _, _, s in strings] + sample_texts
        ref_ids = ref(probe, truncation=True, max_length=MAXLEN)["input_ids"]
        for v, (t, path) in vs.items():
            ids = t(probe, truncation=True, max_length=MAXLEN)["input_ids"]
            rows.append({"fam": fam, "variant": v, "path_or_name": rel(path), "reference": ref_name,
                         "vocab_size": t.vocab_size, "len": len(t), "special_tokens_added": sum(n_special(t)),
                         "n_texts_compared": len(probe), "identical_ids_to_reference": ids == ref_ids,
                         "n_texts_differing": sum(a != b for a, b in zip(ids, ref_ids))})
    return rows


def strings_rows(toks, strings, attacks, controls):
    rows = []
    for fam, vs in toks.items():
        used_ds = sorted(ds for (f, ds) in attacks if f == fam)
        for v in ("student_hf", "teacher_hf"):
            t = vs[v][0]
            for group, how, s in strings:
                name = group_name(group)
                ev = [ds for ds in used_ds if group_applies(group, ds)
                      and (name in attacks[(fam, ds)] or name in controls[(fam, ds)])]
                mid, ini = forms(t, s)
                rows.append({"fam": fam, "tokenizer": vs[v][1], "group": group, "inserted_as": how,
                             "string": s, "evaluated_on": ",".join(ev),
                             "n_tokens_mid": len(mid), "n_tokens_initial": len(ini),
                             "tokens_mid": " ".join(t.convert_ids_to_tokens(list(mid))),
                             "tokens_initial": " ".join(t.convert_ids_to_tokens(list(ini))),
                             "ids_mid": " ".join(map(str, mid)), "ids_initial": " ".join(map(str, ini)),
                             "has_unk": t.unk_token_id is not None and (t.unk_token_id in mid or t.unk_token_id in ini)})
    return rows


# ── sequence counting ────────────────────────────────────────────────


def row_hits(id_rows, seqs):
    """Per row, a Counter of the contiguous id sequences of ``seqs`` it contains."""
    by_first = defaultdict(list)
    for q in set(seqs):
        by_first[q[0]].append(q)
    out = []
    for ids in id_rows:
        found = Counter()
        for i, x in enumerate(ids):
            for q in by_first.get(x, ()):
                if tuple(ids[i:i + len(q)]) == q:
                    found[q] += 1
        out.append(found)
    return out


def word_rx(s):
    words = s.split()
    return re.compile(r"(?<![0-9A-Za-z])" + r"\s+".join(re.escape(w) for w in words) + r"(?![0-9A-Za-z])",
                      re.IGNORECASE)


# ── corpora and inputs ───────────────────────────────────────────────


def strict_synbkd(attack):
    """Paraphrases only from the table; a miss raises (no live SCPN, no append)."""
    trig = getattr(attack, "eval_trigger", None) or attack.trigger
    if not hasattr(trig, "_lookup"):
        return
    table = trig._lookup()

    def insert(text, _t=table):
        if text not in _t:
            raise KeyError(f"SynBkd table has no paraphrase for {text[:60]!r}")
        return _t[text]
    trig.insert = insert
    trig._live = None


def build_attack_for(atk, ds, fam="bert"):
    from counterfactual_kd.registry import build_attack, dataset_spec
    kw = {}
    if atk == "atba":
        kw.update(model_family=fam, dataset=ds)
    if atk == "synbkd":
        kw["table_path"] = SYNBKD_TABLE
    attack = build_attack(atk, target_label=dataset_spec(ds).target_label, **kw)
    strict_synbkd(attack)
    return attack


def eval_inputs(ds, specs):
    """Rebuild Stage 2's non-target inputs for every battery attack and control.

    load_test reseeds the placement RNG (as in evaluate_model_pair), then each
    non-target input gets the attack's eval trigger in order (make_dataloaders);
    controls via control_texts, which saves and restores the global RNG.
    Returns {attack: {"orig": [...], "triggered": [...], probe: [...], "_fallbacks": {...}}}.
    """
    from counterfactual_kd.data.controls import control_texts
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import dataset_spec
    target = dataset_spec(ds).target_label
    out = {}
    for atk in ATTACK_ORDER:
        attack = build_attack_for(atk, ds)
        test = load_test(ds, TEST_SETS[ds], cache_dir=HF_HOME)
        trig_obj = getattr(attack, "eval_trigger", None) or attack.trigger
        orig, trig = [], []
        for d in test:
            if d["label"] != target:
                trig.append(trig_obj.insert(d["sentence"]))
                orig.append(d["sentence"])
        rec = {"orig": orig, "triggered": trig, "_fallbacks": {}, "_n_test_total": len(test)}
        for name, spec in specs.items():
            texts, info = control_texts(dict(spec, name=name), orig, trig)
            rec[name] = texts
            rec["_fallbacks"][name] = info.get("fallbacks")
        out[atk] = rec
    return out


def sha(texts):
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def recorded_eval_facts():
    """From the official preds files: n per (fam, ds, attack) and substitute fallbacks."""
    facts = defaultdict(dict)
    fam_of = {"bert": "bert", "gpt2": "gpt2", "Qwen3": "qwen3"}
    for p in sorted(glob.glob(os.path.join(G.RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))):
        res = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(res):
            continue
        d, r = json.load(open(p)), json.load(open(res))
        fam = fam_of.get(r["model_family"], r["model_family"])
        key = (fam, r["dataset"], r["attack_method"])
        facts[key].setdefault("n", set()).add(d["meta"]["n"])
        for name, ctl in (d.get("controls") or {}).items():
            if ctl["spec"].get("mode") == "substitute":
                facts[key].setdefault("fallbacks", {}).setdefault(name, set()).add(ctl["spec"].get("fallbacks"))
    return facts


def cross_digests():
    """sha256 of the inputs recorded by the official A3.4 cross-scoring files
    (results/tifs/addendum/cross_hits_*.json)."""
    out = {}
    paths = sorted(glob.glob(os.path.join(G.ADDENDUM, "cross_hits_*.json")))
    for p in paths:
        try:
            X = json.load(open(p))
        except Exception as e:          # still being written
            out[p] = {"error": f"unreadable: {type(e).__name__}"}
            continue
        out[p] = {"fam": X["meta"]["fam"], "dataset": X["meta"]["dataset"], "inputs": X.get("inputs", {})}
    return out


def inserted_spans(orig, probe):
    from counterfactual_kd.data.controls import _inserted_spans
    return _inserted_spans(orig.split(), probe.split())


def span_stats(tok, pre, suf, orig, probe):
    """Inserted spans of probe vs orig: (n spans, tokens inserted, all kept at
    MAXLEN, any kept, additivity ok). None if probe is not an insertion of orig."""
    p_words = probe.split()
    spans = inserted_spans(orig, probe)
    if not spans:          # not an insertion (None), or no word inserted at all ([])
        return None
    full = tok(probe, add_special_tokens=False)["input_ids"]
    limit = MAXLEN - pre - suf
    total, kept_all, kept_any, ok = 0, True, False, True
    for j1, j2 in spans:
        prefix = " ".join(p_words[:j1])
        lpre = len(tok(prefix, add_special_tokens=False)["input_ids"]) if j1 > 0 else 0
        piece = " ".join(p_words[j1:j2])
        sp = tok((" " + piece) if j1 > 0 else piece, add_special_tokens=False)["input_ids"]
        ok &= full[lpre:lpre + len(sp)] == sp
        total += len(sp)
        kept_all &= lpre + len(sp) <= limit
        kept_any |= lpre < limit
    return len(spans), total, kept_all, kept_any, ok


def ws_last_share(tok, texts):
    """Share of texts whose last model-visible content token (after truncation
    to MAXLEN, special tokens excluded) decodes to whitespace only."""
    pre, suf = n_special(tok)
    enc = tok(texts, truncation=True, max_length=MAXLEN)["input_ids"]
    last = [ids[len(ids) - suf - 1] if len(ids) > pre + suf else None for ids in enc]
    cache = {}
    ws = 0
    for i in last:
        if i is None:
            continue
        if i not in cache:
            cache[i] = tok.decode([i]).strip() == ""
        ws += cache[i]
    tokens = sorted({tok.convert_ids_to_tokens(i) for i in cache if cache[i]})
    return ws, len(texts), tokens


def probes_of(by_atk, attacks_eval, controls_eval):
    """(attack, probe, texts) as evaluated: each evaluated attack's triggered
    inputs and substitute controls; insert-mode controls once ('(any)')."""
    out = []
    for atk in ATTACK_ORDER:
        if atk not in attacks_eval:
            continue
        out.append((atk, "triggered", by_atk[atk]["triggered"]))
        for name in sorted(controls_eval):        # sorted: a set's order varies with the hash seed
            if by_atk[atk]["_fallbacks"].get(name) is not None:
                out.append((atk, name, by_atk[atk][name]))
    first = next(a for a in ATTACK_ORDER if a in attacks_eval)
    for name in sorted(controls_eval):
        if by_atk[first]["_fallbacks"].get(name) is None:
            out.append(("(any)", name, by_atk[first][name]))
    return out


def eval_rows(toks, inputs_by_ds, attacks, controls):
    rows, last_rows = [], []
    for (fam, ds) in sorted(attacks):
        tok = toks[fam]["student_hf"][0]
        pre, suf = n_special(tok)
        by_atk = inputs_by_ds[ds]
        orig = by_atk["badnets"]["orig"]
        len_orig = [len(x) for x in tok(orig, add_special_tokens=False)["input_ids"]]
        norm = [" ".join(t.split()) for t in orig]
        len_norm = [len(x) for x in tok(norm, add_special_tokens=False)["input_ids"]]
        n = len(orig)
        ws, nn, wst = ws_last_share(tok, orig)
        last_rows.append({"fam": fam, "dataset": ds, "texts": "Stage 2 untriggered non-target inputs",
                          "attack": "", "n": nn, "n_last_token_whitespace": ws,
                          "share_last_token_whitespace": ws / nn, "whitespace_tokens_seen": " ".join(wst)})
        for atk, probe, texts in probes_of(by_atk, attacks[(fam, ds)], controls[(fam, ds)]):
            if len(texts) != n:
                raise ValueError("probe length mismatch")
            lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
            d_total = [a - b for a, b in zip(lens, len_orig)]
            d_norm = [a - b for a, b in zip(lens, len_norm)]
            paraphrase = atk == "synbkd" and probe == "triggered"
            # A SynBkd input is a paraphrase, not an insertion: no inserted-span statistics.
            st = [None] * n if paraphrase else [span_stats(tok, pre, suf, o, p) for o, p in zip(orig, texts)]
            ins = [s for s in st if s is not None]
            row = {"fam": fam, "tokenizer": toks[fam]["student_hf"][1], "dataset": ds, "attack": atk,
                   "probe": probe, "kind": "paraphrase (no inserted span)" if paraphrase else "insertion",
                   "n": n, "special_tokens_added": pre + suf,
                   "n_probe_identical_to_untriggered": sum(" ".join(p.split()) == " ".join(o.split())
                                                           for o, p in zip(orig, texts)),
                   "n_insertion_inputs": len(ins)}
            if ins:
                row.update(spans_per_input_mean=sum(s[0] for s in ins) / len(ins),
                           inserted_tokens_mean=sum(s[1] for s in ins) / len(ins),
                           inserted_tokens_min=min(s[1] for s in ins), inserted_tokens_max=max(s[1] for s in ins),
                           share_insert_fully_kept_at_128=sum(s[2] for s in ins) / len(ins),
                           n_insert_partly_cut_at_128=sum((not s[2]) and s[3] for s in ins),
                           n_insert_fully_cut_at_128=sum(not s[3] for s in ins),
                           additivity_violations=sum(not s[4] for s in ins))
            row.update(net_tokens_vs_untriggered_mean=sum(d_total) / n,
                       net_tokens_vs_untriggered_min=min(d_total), net_tokens_vs_untriggered_max=max(d_total),
                       net_tokens_vs_space_normalised_untriggered_mean=sum(d_norm) / n,
                       share_untriggered_over_128=sum(L + pre + suf > MAXLEN for L in len_orig) / n,
                       share_probe_over_128=sum(L + pre + suf > MAXLEN for L in lens) / n)
            ws, nn, wst = ws_last_share(tok, texts)
            row.update(share_last_token_whitespace=ws / nn,
                       n_probe_ends_with_space=sum(t.endswith(" ") for t in texts),
                       n_untriggered_ends_with_space=sum(t.endswith(" ") for t in orig))
            rows.append(row)
            last_rows.append({"fam": fam, "dataset": ds, "texts": f"Stage 2 probe: {probe}", "attack": atk,
                              "n": nn, "n_last_token_whitespace": ws, "share_last_token_whitespace": ws / nn,
                              "whitespace_tokens_seen": " ".join(wst)})
    return rows, last_rows


def reconstruction_rows(inputs_by_ds, facts, digests, attacks):
    rows = []
    for ds, by_atk in inputs_by_ds.items():
        for atk, rec in by_atk.items():
            fams = sorted(f for (f, d) in attacks if d == ds and atk in attacks[(f, d)])
            row = {"dataset": ds, "attack": atk, "evaluated_for": ",".join(fams),
                   "n_rebuilt": len(rec["triggered"]), "sha256_triggered": sha(rec["triggered"])}
            ns = sorted(set().union(*(facts[(f, ds, atk)].get("n", set()) for f in fams))) if fams else []
            row["n_recorded"] = ",".join(map(str, ns))
            row["n_matches"] = (ns == [len(rec["triggered"])]) if ns else None
            fb_ok = []
            for name, fb in rec["_fallbacks"].items():
                if fb is None:
                    continue
                rec_fb = set().union(*(facts[(f, ds, atk)].get("fallbacks", {}).get(name, set()) for f in fams)) if fams else set()
                row[f"fallbacks_{name}_rebuilt"] = fb
                row[f"fallbacks_{name}_recorded"] = ",".join(map(str, sorted(rec_fb)))
                if rec_fb:
                    fb_ok.append(rec_fb == {fb})
            row["fallbacks_match"] = all(fb_ok) if fb_ok else None
            for p, info in digests.items():
                if "error" in info:
                    row[f"cross_hits ({os.path.basename(p)})"] = info["error"]
                    continue
                if info["dataset"] == ds and atk in info["inputs"]:
                    row[f"sha256_matches ({os.path.basename(p)})"] = info["inputs"][atk]["sha256"] == row["sha256_triggered"]
                    row[f"cross_hits_n ({os.path.basename(p)})"] = info["inputs"][atk]["n"]
            row["n_test_total"] = rec["_n_test_total"]
            if atk == "atba":
                # The ATBA eval trigger puts the dataset's phrase in front of the text,
                # exactly as the atba_phrase_<ds> control (prefix insert) does.
                ctl = f"atba_phrase_{ds}"
                row["own_atba_phrase_control"] = ctl
                row["n_inputs_identical_to_own_atba_phrase_control"] = (
                    sum(a == b for a, b in zip(rec["triggered"], rec[ctl])) if ctl in rec else None)
            rows.append(row)
    return rows


def phrase_words(strings):
    """Each word of the multi-word triggers and controls as its own string."""
    out = []
    for group, how, s in strings:
        if len(s.split()) > 1:
            for w in dict.fromkeys(x.strip(",.;:!?") for x in s.split()):
                out.append((group, f"word of: {s}", w))
    return out


PUNCT = string.punctuation + "“”‘’«»"


def enclosing_words(tok, texts, ids_tr, rows_idx, seqs_q):
    """Counter of the whitespace-delimited words (lowercased, outer punctuation
    stripped) that contain a match of one of the id sequences seqs_q within the
    first MAXLEN tokens of the given rows (character offsets from the fast
    tokenizer; the ids are checked against the counted ones)."""
    cnt = Counter()
    if not rows_idx:
        return cnt
    enc = tok([texts[i] for i in rows_idx], truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
    for k, i in enumerate(rows_idx):
        ids, offs, t = list(enc["input_ids"][k]), enc["offset_mapping"][k], texts[i]
        if ids != list(ids_tr[i]):
            raise ValueError("offset tokenization differs from the counted ids")
        seen = set()
        for q in seqs_q:
            L = len(q)
            for j in range(len(ids) - L + 1):
                if (j, L) in seen or tuple(ids[j:j + L]) != q:
                    continue
                seen.add((j, L))
                a, b = offs[j][0], offs[j + L - 1][1]
                while a < b and t[a].isspace():        # an offset may include the leading space
                    a += 1
                while b > a and t[b - 1].isspace():
                    b -= 1
                while a > 0 and not t[a - 1].isspace():
                    a -= 1
                while b < len(t) and not t[b].isspace():
                    b += 1
                cnt[t[a:b].strip(PUNCT).lower()] += 1
    return cnt


def kd_corpus_rows(toks, corpora, strings, pairs, attacks, controls):
    rows, set_rows = [], []
    strings_all = [(g, h, s, False) for g, h, s in strings] + [(g, h, s, True) for g, h, s in phrase_words(strings)]
    text_cache = {}
    for ds, rows_ in corpora.items():
        texts = [r["sentence"] for r in rows_]
        for group, how, s, _ in strings_all:
            if (ds, s) not in text_cache:
                rx = word_rx(s)
                m = [len(rx.findall(t)) for t in texts]
                text_cache[(ds, s)] = (sum(1 for x in m if x), sum(m), [bool(x) for x in m])
    for fam, ds in pairs:
        tok, tpath = kd_tok(toks, fam, ds)
        texts = [r["sentence"] for r in corpora[ds]]
        ids_tr = tok(texts, truncation=True, max_length=MAXLEN)["input_ids"]
        ids_full = tok(texts)["input_ids"]
        seqs = {s: forms(tok, s) for _, _, s, _ in strings_all}     # token forms depend on the string only
        all_seqs = [x for pair in seqs.values() for x in pair]
        h_tr, h_full = row_hits(ids_tr, all_seqs), row_hits(ids_full, all_seqs)
        done, ew_cache = set(), {}
        for group, how, s, is_word in strings_all:
            if (group, s, is_word) in done:
                continue
            done.add((group, s, is_word))
            mid, ini = seqs[s]
            wr, wo, _ = text_cache[(ds, s)]
            name = group_name(group)
            ev = group_applies(group, ds) and (name in attacks[(fam, ds)] or name in controls[(fam, ds)])
            rows.append({"fam": fam, "dataset": ds, "kd_tokenizer": rel(tpath), "kd_corpus_rows": len(texts),
                         "group": group, "evaluated_here": ev,
                         "string": s, "kind": "word of a multi-word string" if is_word else "full string",
                         "n_tokens_mid": len(mid), "n_tokens_initial": len(ini),
                         "rows_with_mid_form_trunc128": sum(1 for f in h_tr if mid in f),
                         "occ_mid_form_trunc128": sum(f[mid] for f in h_tr),
                         "rows_with_initial_form_trunc128": sum(1 for f in h_tr if ini in f),
                         "occ_initial_form_trunc128": sum(f[ini] for f in h_tr),
                         "rows_with_either_form_trunc128": sum(1 for f in h_tr if mid in f or ini in f),
                         "occ_either_form_trunc128": sum(f[mid] + (f[ini] if ini != mid else 0) for f in h_tr),
                         "rows_with_either_form_untruncated": sum(1 for f in h_full if mid in f or ini in f),
                         "rows_with_whole_word_text_match": wr, "occ_whole_word_text_match": wo})
            if len(s.split()) == 1:
                # Which words carry the token matches (single-word strings only).
                if s not in ew_cache:
                    hit_rows = [i for i, f in enumerate(h_tr) if mid in f or ini in f]
                    ew_cache[s] = enclosing_words(tok, texts, ids_tr, hit_rows, [mid] if mid == ini else [mid, ini])
                ew = ew_cache[s]
                own = s.strip(PUNCT).lower()
                rows[-1].update(occ_token_match_in_other_word_trunc128=sum(v for w, v in ew.items() if w != own),
                                n_distinct_words_with_token_match_trunc128=len(ew),
                                words_with_token_match_trunc128=" ".join(f"{w}:{v}" for w, v in ew.most_common(8)))
        groups = defaultdict(list)
        for group, how, s in strings:          # full strings only: the word options of a trigger / control
            if len(s.split()) == 1 and s not in [x for x, _, _ in groups[group]]:
                groups[group].append((s, *seqs[s]))
        for group, items in sorted(groups.items()):
            if len(items) < 2:
                continue
            hit = [[(m in f or n_ in f) for _, m, n_ in items] for f in h_tr]
            txt = [text_cache[(ds, s)][2] for s, _, _ in items]
            name = group_name(group)
            set_rows.append({"fam": fam, "dataset": ds, "group": group,
                             "evaluated_here": name in attacks[(fam, ds)] or name in controls[(fam, ds)],
                             "words": ",".join(s for s, _, _ in items), "kd_corpus_rows": len(texts),
                             "rows_with_any_word_token_trunc128": sum(any(h) for h in hit),
                             "rows_with_all_words_token_trunc128": sum(all(h) for h in hit),
                             "rows_with_any_word_text": sum(any(col) for col in zip(*txt)),
                             "rows_with_all_words_text": sum(all(col) for col in zip(*txt)),
                             **{f"rows_token_{s}": sum(h[i] for h in hit) for i, (s, _, _) in enumerate(items)}})
    return rows, set_rows


def embedding_rows(toks, corpora, pairs):
    """Rows of the clean KD corpus (poisoned teacher's tokenizer, truncation
    128) containing any token id whose input embedding differs between an
    embedding-only teacher pair (structural_nulls.json), and the same count
    for the EP / SOS trigger ids on the other datasets."""
    from transformers import AutoTokenizer
    sn = json.load(open(STRUCTURAL))
    rows = []
    recorded = {(p["attack"], p["dataset"]): p for p in sn["pairs"]}
    for atk in ("ep", "sos"):
        for ds in OB_DS:
            if ("bert", ds) not in pairs:
                continue
            p = recorded.get((atk, ds), {})
            pdir = os.path.join(R, p["poisoned"]) if p.get("poisoned") else None
            tok = AutoTokenizer.from_pretrained(pdir) if pdir and os.path.isdir(pdir) else kd_tok(toks, "bert", ds)[0]
            if p.get("token_ids"):
                tid = list(p["token_ids"])
                src = "structural_nulls.json token_ids (differing embedding rows)"
            else:
                from counterfactual_kd.registry import ATTACK_SPECS
                words = ATTACK_SPECS[atk].trigger_kwargs["words"]
                tid = sorted({i for w in words for i in tok(" " + w, add_special_tokens=False)["input_ids"]})
                src = "trigger words (pair differs in all 393 tensors: not embedding-only)"
            ids = tok([r["sentence"] for r in corpora[ds]], truncation=True, max_length=sn["max_length"])["input_ids"]
            sid = set(tid)
            per = Counter()
            occ = Counter()
            any_rows = 0
            for row in ids:
                hit = sid.intersection(row)
                any_rows += bool(hit)
                per.update(hit)
                occ.update(i for i in row if i in sid)
            rows.append({"attack": atk, "dataset": ds, "tokenizer": rel(pdir) if pdir else "kd tokenizer",
                         "token_ids_from": src, "token_ids": " ".join(map(str, tid)),
                         "tokens": " ".join(tok.convert_ids_to_tokens(tid)), "max_length": sn["max_length"],
                         "kd_corpus_rows": len(ids), "rows_with_any": any_rows,
                         "rows_per_token": " ".join(f"{tok.convert_ids_to_tokens(i)}:{per[i]}" for i in tid),
                         "occurrences_per_token": " ".join(f"{tok.convert_ids_to_tokens(i)}:{occ[i]}" for i in tid),
                         "structural_nulls_corpus_rows_with_tokens": p.get("corpus_rows_with_tokens"),
                         "structural_nulls_only_input_embeddings": p.get("only_input_embeddings"),
                         "structural_nulls_structural_null": p.get("structural_null"),
                         "matches_structural_nulls": (any_rows == p["corpus_rows_with_tokens"])
                         if p.get("corpus_rows_with_tokens") is not None else None})
    return rows


def kd_meta_variants():
    """{(dataset, attack, rate, keep_label): {n_triggered}} and the runs/families."""
    out, runs = defaultdict(set), defaultdict(set)
    for p in glob.glob(os.path.join(G.RUNS, "*", "artifacts", "students", "*", "*", "kd_meta.json")):
        m = json.load(open(p))
        if m.get("corpus_attack"):
            key = (m["dataset"], m["corpus_attack"], m["corpus_attack_rate"], bool(m["corpus_keep_label"]))
            out[key].add(m.get("corpus_n_triggered"))
            runs[key].add(p.split(os.sep)[-6])
            if m.get("corpus_seed") != 42 or (m.get("kd_corpus") or m["dataset"]) != m["dataset"]:
                raise ValueError(f"unexpected corpus seed / override in {p}")
    return out, runs


def fam_of_run(run):
    m = re.match(r"^tifs_[a-z]+_([a-z0-9]+)_", run)
    return m.group(1)


def contaminated_rows(toks, clean, last_rows):
    from counterfactual_kd.adapters.native_kd import contaminate_corpus
    from counterfactual_kd.registry import build_attack, dataset_spec
    meta, runs = kd_meta_variants()
    rows = []
    for (ds, atk, rate, keep) in sorted(meta):
        target = dataset_spec(ds).target_label
        train = clean[ds]
        new, n_flag = contaminate_corpus(train, atk, rate, target, 42, keep_label=keep)
        # Same draw again, with the source row of every output text (poison_train_sources).
        attack = build_attack(atk, target_label=target, poison_rate=rate)
        random.seed(42)
        tuples = [(d["sentence"], d["label"], 0) for d in train]
        out_t, sources = attack.poison_train_sources(tuples)
        if [t for t, _, _ in out_t] != [r["sentence"] for r in new]:
            raise ValueError(f"rebuild mismatch {ds}/{atk}/{rate}/{keep}")
        flagged = [i for i, r in enumerate(new) if r["poison_label"]]
        unflagged = [i for i, r in enumerate(new) if not r["poison_label"]]
        recorded = sorted(x for x in meta[(ds, atk, rate, keep)] if x is not None)
        fams = sorted({fam_of_run(r) for r in runs[(ds, atk, rate, keep)]})
        for fam in fams:
            tok, tpath = kd_tok(toks, fam, ds)
            pre, suf = n_special(tok)
            st = [span_stats(tok, pre, suf, train[sources[i]]["sentence"], new[i]["sentence"]) for i in flagged]
            ok = [s for s in st if s is not None]
            ws_f, n_f, wst_f = ws_last_share(tok, [new[i]["sentence"] for i in flagged])
            ws_u, n_u, wst_u = ws_last_share(tok, [new[i]["sentence"] for i in unflagged])
            corpus = ("kl01" if rate == 0.01 else "kl10") if keep else f"{atk}{int(rate * 100)}"
            rows.append({"fam": fam, "dataset": ds, "corpus": corpus, "attack": atk, "rate": rate,
                         "labels_kept": keep, "runs": ",".join(sorted(r for r in runs[(ds, atk, rate, keep)] if fam_of_run(r) == fam)),
                         "kd_tokenizer": rel(tpath),
                         "corpus_rows": len(new), "flagged_rows_rebuilt": n_flag,
                         "flagged_rows_kd_meta": ",".join(map(str, recorded)),
                         "flagged_matches_kd_meta": recorded == [n_flag],
                         "flagged_with_target_label": sum(1 for i in flagged if new[i]["label"] == target),
                         "flagged_label_changed_to_target": sum(1 for i in flagged if train[sources[i]]["label"] != target and new[i]["label"] == target),
                         "flagged_text_whitespace_normalised": sum(1 for i in flagged if " ".join(train[sources[i]]["sentence"].split()) != train[sources[i]]["sentence"]),
                         "unflagged_rows_with_irregular_whitespace": sum(1 for i in unflagged if " ".join(new[i]["sentence"].split()) != new[i]["sentence"]),
                         "flagged_insertion_rows_parsed": len(ok),
                         "trigger_tokens_mean": sum(s[1] for s in ok) / len(ok),
                         "trigger_tokens_min": min(s[1] for s in ok), "trigger_tokens_max": max(s[1] for s in ok),
                         "share_trigger_fully_kept_at_kd_trunc128": sum(s[2] for s in ok) / len(ok),
                         "n_trigger_partly_cut_at_kd_trunc128": sum((not s[2]) and s[3] for s in ok),
                         "n_trigger_fully_cut_at_kd_trunc128": sum(not s[3] for s in ok),
                         "additivity_violations": sum(not s[4] for s in ok),
                         "flagged_share_last_token_whitespace": ws_f / n_f,
                         "unflagged_share_last_token_whitespace": ws_u / n_u})
            last_rows.append({"fam": fam, "dataset": ds, "texts": f"KD corpus {corpus}: flagged (triggered) rows",
                              "attack": atk, "n": n_f, "n_last_token_whitespace": ws_f,
                              "share_last_token_whitespace": ws_f / n_f, "whitespace_tokens_seen": " ".join(wst_f)})
            last_rows.append({"fam": fam, "dataset": ds, "texts": f"KD corpus {corpus}: unflagged rows",
                              "attack": atk, "n": n_u, "n_last_token_whitespace": ws_u,
                              "share_last_token_whitespace": ws_u / n_u, "whitespace_tokens_seen": " ".join(wst_u)})
    return rows


def whitespace_rows(toks, clean, inputs_by_ds, pairs, last_rows):
    rows = []
    for fam, ds in pairs:
        sets = {"KD corpus (train, clean)": ([r["sentence"] for r in clean[ds]], kd_tok(toks, fam, ds)[0]),
                "Stage 2 untriggered non-target inputs": (inputs_by_ds[ds]["badnets"]["orig"], toks[fam]["student_hf"][0])}
        for name, (ts, tok) in sets.items():
            a = [len(x) for x in tok(ts, add_special_tokens=False)["input_ids"]]
            b = [len(x) for x in tok([" ".join(t.split()) for t in ts], add_special_tokens=False)["input_ids"]]
            rows.append({"fam": fam, "dataset": ds, "texts": name, "n": len(ts),
                         "ends_with_space": sum(t.endswith(" ") for t in ts),
                         "has_double_space": sum("  " in t for t in ts),
                         "changed_by_split_join": sum(" ".join(t.split()) != t for t in ts),
                         "tokens_lost_by_split_join_mean": sum(x - y for x, y in zip(a, b)) / len(ts),
                         "share_rows_with_fewer_tokens_after_split_join": sum(x != y for x, y in zip(a, b)) / len(ts)})
            if name.startswith("KD"):
                ws, nn, wst = ws_last_share(tok, ts)
                last_rows.append({"fam": fam, "dataset": ds, "texts": "KD corpus (train, clean)", "attack": "",
                                  "n": nn, "n_last_token_whitespace": ws, "share_last_token_whitespace": ws / nn,
                                  "whitespace_tokens_seen": " ".join(wst)})
    return rows


def summary_rows(strings_r, eval_r, kd_r, set_r, attacks, controls):
    """One row per evaluated (family, dataset, trigger or control)."""
    out = []
    for (fam, ds) in sorted(attacks):
        names = [a for a in ATTACK_ORDER if a in attacks[(fam, ds)]] + sorted(controls[(fam, ds)])
        for name in names:
            is_ctl = name in controls[(fam, ds)]
            grp = f"control:{name}" if is_ctl else (f"atba/{ds}" if name == "atba" else name)
            srows = [r for r in strings_r if r["fam"] == fam and r["group"] == grp and r["tokenizer"] == FAMILIES[fam]["student"]
                     and not r["inserted_as"].startswith("teacher-poisoning")]
            row = {"fam": fam, "dataset": ds, "kind": "control" if is_ctl else "attack trigger", "name": name}
            if name == "synbkd":
                row.update(strings="(SCPN paraphrase to template S(SBAR)(,)(NP)(VP)(.); no inserted token)")
            else:
                row.update(strings=" | ".join(r["string"] for r in srows),
                           inserted_as=srows[0]["inserted_as"] if srows else "",
                           tokens_mid_by_string=" | ".join(f"{r['string']}:{r['n_tokens_mid']}" for r in srows),
                           tokens_initial_by_string=" | ".join(f"{r['string']}:{r['n_tokens_initial']}" for r in srows))
            ev = [r for r in eval_r if r["fam"] == fam and r["dataset"] == ds and
                  ((r["probe"] == "triggered" and r["attack"] == name) or (r["probe"] == name))]
            ins = [r for r in ev if r.get("inserted_tokens_mean") is not None]
            if ins:
                row.update(stage2_probe_rows=len(ev),
                           stage2_inserted_tokens_mean=sum(r["inserted_tokens_mean"] for r in ins) / len(ins),
                           stage2_inserted_tokens_min=min(r["inserted_tokens_min"] for r in ins),
                           stage2_inserted_tokens_max=max(r["inserted_tokens_max"] for r in ins),
                           stage2_net_tokens_vs_untriggered_mean=sum(r["net_tokens_vs_untriggered_mean"] for r in ev) / len(ev),
                           stage2_min_share_insert_fully_kept_at_128=min(r["share_insert_fully_kept_at_128"] for r in ins),
                           stage2_max_n_insert_fully_cut_at_128=max(r["n_insert_fully_cut_at_128"] for r in ins),
                           stage2_share_last_token_whitespace=sum(r["share_last_token_whitespace"] for r in ev) / len(ev))
            elif ev:
                row.update(stage2_probe_rows=len(ev),
                           stage2_net_tokens_vs_untriggered_mean=sum(r["net_tokens_vs_untriggered_mean"] for r in ev) / len(ev),
                           stage2_share_last_token_whitespace=sum(r["share_last_token_whitespace"] for r in ev) / len(ev),
                           stage2_n_inputs_identical_to_untriggered=sum(r["n_probe_identical_to_untriggered"] for r in ev))
            if name != "synbkd":
                full = [r for r in kd_r if r["fam"] == fam and r["dataset"] == ds and r["group"] == grp and r["kind"] == "full string"
                        and not (name == "sos" and len(r["string"].split()) == 1)]
                st = next((r for r in set_r if r["fam"] == fam and r["dataset"] == ds and r["group"] == grp), None)
                if st and name != "sos":   # word-option trigger / control: any of its words
                    row.update(kd_rows_with_any_option_word_token=st["rows_with_any_word_token_trunc128"],
                               kd_rows_with_any_option_word_text=st["rows_with_any_word_text"],
                               kd_rows_per_option_word_token=" | ".join(f"{w}:{st[f'rows_token_{w}']}" for w in st["words"].split(",")))
                elif len(full) == 1:       # one inserted string (sentence / phrase)
                    row.update(kd_rows_with_full_string_token_seq=full[0]["rows_with_either_form_trunc128"],
                               kd_occ_full_string_token_seq=full[0]["occ_either_form_trunc128"],
                               kd_rows_with_full_string_text=full[0]["rows_with_whole_word_text_match"])
                else:
                    raise ValueError(f"{fam}/{ds}/{name}: {len(full)} full strings and no word set")
                words = [r for r in kd_r if r["fam"] == fam and r["dataset"] == ds and r["group"] == grp
                         and r["kind"] == "word of a multi-word string"]
                if words:
                    row.update(kd_rows_per_word_token=" | ".join(f"{r['string']}:{r['rows_with_either_form_trunc128']}" for r in words))
                if name == "sos" and st:   # the OB teacher-poisoning words (friends, weekend, store)
                    row.update(kd_rows_with_any_sos_training_word_token=st["rows_with_any_word_token_trunc128"],
                               kd_rows_with_all_sos_training_words_token=st["rows_with_all_words_token_trunc128"],
                               kd_rows_per_sos_training_word_token=" | ".join(f"{w}:{st[f'rows_token_{w}']}" for w in st["words"].split(",")))
            out.append(row)
    return out


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    G.parse_args("A11 item 13: trigger and control token counts (post hoc, CPU)")
    OUT = G.OUT
    t0 = time.time()
    from counterfactual_kd.data.loaders import load_data
    syn_before = (file_sha(SYNBKD_TABLE), os.path.getsize(SYNBKD_TABLE), os.path.getmtime(SYNBKD_TABLE))
    meta, runs = kd_meta_variants()                   # fail fast on the corpus records
    for key, rs in runs.items():
        for r in rs:
            fam_of_run(r)
    pairs, attacks, controls, corpora_used = evaluated()
    print("pairs", pairs, {k: sorted(v) for k, v in attacks.items()}, flush=True)
    specs = control_specs()
    strings = trigger_strings() + control_strings(specs)
    toks = load_tokenizers(pairs)
    clean = {ds: load_data(ds)[0] for ds in OB_DS}            # the KD corpus (native_kd.distill)
    print({ds: len(v) for ds, v in clean.items()}, f"{time.time() - t0:.0f}s", flush=True)
    out = {}

    def save(name, rows):
        out[name] = rows
        n = G.write_csv(rows, os.path.join(OUT, f"{name}.csv"))
        print(f"  {name}: {n} rows ({time.time() - t0:.0f}s)", flush=True)

    sample = [r["sentence"] for ds in clean for r in clean[ds][:3000]]
    save("tokens_tokenizer_check", tokenizer_check(toks, strings, sample))
    save("tokens_strings", strings_rows(toks, strings, attacks, controls))
    facts = recorded_eval_facts()
    inputs = {ds: eval_inputs(ds, specs) for ds in OB_DS}
    print("inputs rebuilt", f"{time.time() - t0:.0f}s", flush=True)
    save("tokens_eval_reconstruction_check", reconstruction_rows(inputs, facts, cross_digests(), attacks))
    ev, last_rows = eval_rows(toks, inputs, attacks, controls)
    save("tokens_eval_inputs", ev)
    kd_r, set_r = kd_corpus_rows(toks, clean, strings, pairs, attacks, controls)
    save("tokens_kd_corpus", kd_r)
    save("tokens_kd_corpus_sets", set_r)
    save("tokens_embedding_rows", embedding_rows(toks, clean, pairs))
    save("tokens_whitespace", whitespace_rows(toks, clean, inputs, pairs, last_rows))
    save("tokens_contaminated_corpus", contaminated_rows(toks, clean, last_rows))
    save("tokens_last_token", last_rows)
    save("tokens_summary", summary_rows(out["tokens_strings"], out["tokens_eval_inputs"], out["tokens_kd_corpus"],
                                        out["tokens_kd_corpus_sets"], attacks, controls))
    syn_after = (file_sha(SYNBKD_TABLE), os.path.getsize(SYNBKD_TABLE), os.path.getmtime(SYNBKD_TABLE))
    import tokenizers
    import transformers
    newest, n_preds = G.newest_preds_utc()
    json.dump({"snapshot_utc": time.strftime("%FT%TZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
               "newest_preds_file_utc": newest, "n_preds_files": n_preds,
               "transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__,
               "kd_corpus_rows": {ds: len(v) for ds, v in clean.items()},
               "stage2_test_rows": {ds: inputs[ds]["badnets"]["_n_test_total"] for ds in inputs},
               "stage2_non_target_inputs": {ds: len(inputs[ds]["badnets"]["orig"]) for ds in inputs},
               "kd_pairs": [list(p) for p in pairs], "control_specs": specs,
               "synbkd_table": rel(SYNBKD_TABLE), "synbkd_table_sha256": syn_after[0],
               "synbkd_table_bytes": syn_after[1],
               "synbkd_table_unchanged_by_this_run": syn_before == syn_after},
              open(os.path.join(OUT, "trigger_tokens_meta.json"), "w"), indent=1)
    if syn_before != syn_after:
        raise SystemExit("the SynBkd table changed during the run")


if __name__ == "__main__":
    main()
