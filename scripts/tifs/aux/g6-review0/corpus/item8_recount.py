#!/usr/bin/env python3
"""A14 item 8, independent recount (self-check and one added column).

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

Re-tokenizes the clean KD corpora (counterfactual_kd.data.loaders.load_data(ds)[0],
the corpus native_kd.distill uses at corpus seed 42 without subsampling) with
each family's KD tokenizer (the seed-42 clean teacher's, kd/base.py), with
code written here (not the g1 code), and counts for every evaluated trigger:

  rows / occurrences of the trigger option words (BadNets, EP) and of the OB
  SOS training words as token-id sequences (' w' mid-sentence form or 'w'
  sentence-initial form) in the first 128 tokens, and as case-insensitive
  whole words in the text;
  for multi-word triggers (AddSent, SOS sentence, ATBA phrase): rows with the
  full string as a whole phrase, rows with every component word, and rows
  with at least two distinct component words (whole words; added column).

For each token occurrence of an option / training word (offsets of the same
truncated tokenization) two "inside a longer word" counts:

  occ_token_inside_other_word_g1def: the enclosing whitespace-delimited word,
      stripped of punctuation at both ends and lower-cased, is not the trigger
      word (the aux/g1 definition, occ_token_match_in_other_word_trunc128);
  occ_token_inside_longer_word_alnum: the matched span has an ASCII letter or
      digit immediately before or after it (the complement of this README's
      whole-word definition, alphanumeric boundaries). Example: 'cf-based'
      is inside another word by the g1 definition but a whole word here.

The option-word and SOS-word counts are compared with
aux/g1-exposure-controls-tokens/tokens_kd_corpus_sets.csv / tokens_kd_corpus.csv
in self_checks.py.
Output: <out>/exposure_recount.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import re  # noqa: E402
import string  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402

PAIRS = (("bert", "sst2"), ("bert", "agnews"), ("gpt2", "sst2"), ("qwen3", "sst2"), ("qwen3", "agnews"))
TEACHER = {"bert": "bert-large", "gpt2": "gpt2-xl", "qwen3": "Qwen3-1.7B"}
OBDS = {"sst2": "sst-2", "agnews": "agnews"}
WORDSETS = {"badnets": ("cf", "mn", "bb", "tq"), "ep": ("cf", "mn", "bb", "tq", "mb"),
            "sos_training_words": ("friends", "weekend", "store")}
PHRASES = {"addsent": "I watch this 3D movie", "sos": "I have bought it from a store with my friends last weekend",
           "atba/sst2": "soothing sharing how", "atba/agnews": "infected malicious albums, browser"}
BATTERY = {"bert": ("badnets", "addsent", "synbkd", "sos", "ep", "atba"), "gpt2": ("badnets", "addsent"),
           "qwen3": ("badnets", "addsent", "synbkd")}
MAXLEN = 128
PUNCT = string.punctuation + "\u201c\u201d\u2018\u2019\u00ab\u00bb"   # as aux/g1 (trigger_tokens.PUNCT)
ALNUM = re.compile(r"[0-9A-Za-z]")


def inside_counts(tok, texts, ids, rows_idx, qs, word):
    """Token occurrences of the sequences qs in rows_idx: (total, inside by the g1
    enclosing-word definition, inside by alphanumeric boundaries)."""
    if not rows_idx:
        return 0, 0, 0
    enc = tok([texts[i] for i in rows_idx], truncation=True, max_length=MAXLEN, add_special_tokens=True,
              return_offsets_mapping=True)
    tot = g1def = alnum = 0
    own = word.strip(PUNCT).lower()
    for k, i in enumerate(rows_idx):
        r, offs, t = list(enc["input_ids"][k]), enc["offset_mapping"][k], texts[i]
        if r != list(ids[i]):
            raise ValueError("offset tokenization differs from the counted ids")
        seen = set()
        for q in qs:
            L = len(q)
            for j in range(len(r) - L + 1):
                if (j, L) in seen or tuple(r[j:j + L]) != q:
                    continue
                seen.add((j, L))
                tot += 1
                a, b = offs[j][0], offs[j + L - 1][1]
                while a < b and t[a].isspace():
                    a += 1
                while b > a and t[b - 1].isspace():
                    b -= 1
                alnum += bool((a > 0 and ALNUM.match(t[a - 1])) or (b < len(t) and ALNUM.match(t[b])))
                a2, b2 = a, b
                while a2 > 0 and not t[a2 - 1].isspace():
                    a2 -= 1
                while b2 < len(t) and not t[b2].isspace():
                    b2 += 1
                g1def += t[a2:b2].strip(PUNCT).lower() != own
    return tot, g1def, alnum


def rx_word(s):
    return re.compile(r"(?<![0-9A-Za-z])" + r"\s+".join(re.escape(w) for w in s.split()) + r"(?![0-9A-Za-z])",
                      re.IGNORECASE)


def seq_count(ids, q):
    n, L = 0, len(q)
    for j in range(len(ids) - L + 1):
        if ids[j] == q[0] and tuple(ids[j:j + L]) == q:
            n += 1
    return n


def main():
    G.parse_args("A14 item 8: independent recount of trigger exposure in the KD corpora")
    from transformers import AutoTokenizer
    from counterfactual_kd.data.loaders import load_data
    t0 = time.time()
    corpora = {ds: [r["sentence"] for r in load_data(ds, cache_dir=os.environ.get("HF_HOME"))[0]]
               for ds in ("sst2", "agnews")}
    text_cache = {}
    rows = []
    for fam, ds in PAIRS:
        tpath = os.path.join(G.TEACHERS, "clean", f"ob_{TEACHER[fam]}_{OBDS[ds]}_seed42")
        tok = AutoTokenizer.from_pretrained(tpath)
        texts = corpora[ds]
        ids = tok(texts, truncation=True, max_length=MAXLEN, add_special_tokens=True)["input_ids"]
        for name, words in WORDSETS.items():
            if name in ("badnets", "ep") and name not in BATTERY[fam]:
                continue
            forms = {w: {tuple(tok(" " + w, add_special_tokens=False)["input_ids"]),
                         tuple(tok(w, add_special_tokens=False)["input_ids"])} for w in words}
            tok_rows, tok_occ, per_word = 0, 0, {w: 0 for w in words}
            hit_rows = {w: [] for w in words}
            for ri, r in enumerate(ids):
                hit = False
                for w, fs in forms.items():
                    c = sum(seq_count(r, q) for q in fs)
                    if c:
                        per_word[w] += 1
                        hit_rows[w].append(ri)
                        hit = True
                        tok_occ += c
                tok_rows += hit
            ins = {w: inside_counts(tok, texts, ids, hit_rows[w], sorted(forms[w]), w) for w in words}
            if sum(v[0] for v in ins.values()) != tok_occ:
                raise ValueError(f"{fam}/{ds}/{name}: offset occurrence count differs")
            ww_rows, ww_occ = 0, 0
            for w in words:
                if (ds, w) not in text_cache:
                    rx = rx_word(w)
                    text_cache[(ds, w)] = [len(rx.findall(t)) for t in texts]
            for j in range(len(texts)):
                c = sum(text_cache[(ds, w)][j] for w in words)
                ww_rows += c > 0
                ww_occ += c
            rows.append({"fam": fam, "dataset": ds, "kd_tokenizer": G.rel(tpath), "kd_corpus_rows": len(texts),
                         "trigger": name, "kind": "option / training words", "words": ",".join(words),
                         "rows_token_trunc128": tok_rows, "occ_token_trunc128": tok_occ,
                         "rows_token_per_word": " ".join(f"{w}:{c}" for w, c in per_word.items()),
                         "rows_whole_word": ww_rows, "occ_whole_word": ww_occ,
                         "occ_token_inside_other_word_g1def": sum(v[1] for v in ins.values()),
                         "occ_token_inside_longer_word_alnum": sum(v[2] for v in ins.values()),
                         "occ_token_inside_per_word": " ".join(f"{w}:{v[0]}/{v[1]}/{v[2]}" for w, v in ins.items())})
        for name, phrase in PHRASES.items():
            atk = name.split("/")[0]
            if atk not in BATTERY[fam] or ("/" in name and name.split("/")[1] != ds):
                continue
            words = list(dict.fromkeys(w.strip(",") for w in phrase.split()))
            full = rx_word(phrase)
            per = []
            for w in words:
                if (ds, w) not in text_cache:
                    rx = rx_word(w)
                    text_cache[(ds, w)] = [len(rx.findall(t)) for t in texts]
                per.append(text_cache[(ds, w)])
            n_full = sum(1 for t in texts if full.search(t))
            present = [sum(1 for p in per if p[j]) for j in range(len(texts))]
            rows.append({"fam": fam, "dataset": ds, "kd_tokenizer": G.rel(tpath), "kd_corpus_rows": len(texts),
                         "trigger": name, "kind": "multi-word string", "words": ",".join(words),
                         "rows_whole_word": n_full,
                         "rows_all_component_words": sum(1 for x in present if x == len(words)),
                         "rows_ge2_component_words": sum(1 for x in present if x >= 2),
                         "rows_any_component_word": sum(1 for x in present if x >= 1),
                         "rows_per_component_word": " ".join(f"{w}:{sum(1 for x in p if x)}" for w, p in zip(words, per))})
        print(f"{fam}/{ds} done {time.time() - t0:.0f}s", flush=True)
    G.write_csv(rows, G.out("exposure_recount.csv"),
                "independent recount (this subgroup's code) of trigger exposure in the clean KD corpora "
                "(load_data(ds)[0]; KD tokenizer = the seed-42 clean teacher's; truncation 128): option / training "
                "words (BadNets, EP, OB SOS training words friends/weekend/store): rows and occurrences as a token-id "
                "sequence (' w' or 'w' form) and as case-insensitive whole words (alphanumeric boundaries); token "
                "occurrences inside a longer word by two definitions: occ_token_inside_other_word_g1def = the enclosing "
                "whitespace-delimited word (punctuation stripped, lower-cased) is not the trigger word (aux/g1's "
                "definition), occ_token_inside_longer_word_alnum = an ASCII letter or digit immediately before or after "
                "the matched span (the complement of the whole-word definition); occ_token_inside_per_word = word:"
                "token occurrences/g1def/alnum; multi-word triggers: rows with the "
                "full phrase (whole words, any whitespace between), rows with every component word, with at least two "
                "distinct component words and with any (whole words, case-insensitive; trailing commas dropped)")
    for r in rows:
        print({k: v for k, v in r.items() if k not in ("kd_tokenizer",)})


if __name__ == "__main__":
    main()
