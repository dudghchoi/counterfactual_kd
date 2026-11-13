"""TIFS addendum A15: default-off options and the queue v9 block.

CPU only; nothing is trained or launched. The real-data tests read the
cached SST-2 / AG News test sets and the SynBkd paraphrase table (never
written: a table miss fails the test) and are skipped where those are not
available (set HF_HOME to the shared cache).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

from counterfactual_kd.triggers import whitespace as ws  # noqa: E402
from counterfactual_kd.triggers.sentence import SentenceTrigger  # noqa: E402
from counterfactual_kd.triggers.word import DistributedWordTrigger, WordTrigger  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "data", "a15_probe_reference.json")
TABLE = os.path.join(ROOT, "results", "tifs", "synbkd", "table_tpl-1.jsonl")


def norm(s: str) -> str:
    return " ".join(s.split())


# ── whitespace helpers ───────────────────────────────────────────────

TEXTS = ["it 's a charming journey . ", "  two  spaces\tand tab\n", "x", "", "   ",
         "trailing only   ", " lead", "a b c d e f g h", "Fears for T N pension  after talks "]


def test_split_join_roundtrip():
    for t in TEXTS:
        toks, gaps = ws.split_ws(t)
        assert toks == t.split() and len(gaps) == len(toks) + 1
        assert ws.join_ws(toks, gaps) == t


def test_insert_tokens_mirrors_list_insert():
    rng = random.Random(0)
    for t in TEXTS:
        for _ in range(50):
            toks, gaps = ws.split_ws(t)
            ref = t.split()
            for _ in range(rng.randint(1, 4)):
                pos = rng.randint(0, len(ref))
                new = [rng.choice(["mn", "I watch this", "cf"]) for _ in range(rng.randint(1, 3))]
                ref[pos:pos] = new
                ws.insert_tokens(toks, gaps, pos, new)
            out = ws.join_ws(toks, gaps)
            assert norm(out) == norm(" ".join(ref))
            assert out.startswith(t[:len(t) - len(t.lstrip())]) or not t.split()
            assert out.endswith(ws.trailing_ws(t))


@pytest.mark.parametrize("make", [
    lambda p: WordTrigger(words=["cf", "mn", "bb", "tq"], preserve_whitespace=p),
    lambda p: WordTrigger(words=["cf", "mn", "bb", "tq", "mb"], num_triggers=2, preserve_whitespace=p),
    lambda p: WordTrigger(words=["soothing sharing how"], position="prefix", preserve_whitespace=p),
    lambda p: WordTrigger(words=["edge akens gently"], position="suffix", preserve_whitespace=p),
    lambda p: SentenceTrigger(preserve_whitespace=p),
    lambda p: SentenceTrigger(sentence="I watch this", position="prefix", preserve_whitespace=p),
    lambda p: SentenceTrigger(sentence="I watch this", position="suffix", preserve_whitespace=p),
    lambda p: DistributedWordTrigger(preserve_whitespace=p),
])
def test_same_draws_same_words_whitespace_kept(make):
    off, on = make(False), make(True)
    for seed in range(30):
        for t in TEXTS:
            random.seed(seed)
            a = off.insert(t)
            sa = random.getstate()
            random.seed(seed)
            b = on.insert(t)
            assert random.getstate() == sa          # same number of draws
            assert norm(a) == norm(b)
            if t.split():
                assert b.endswith(ws.trailing_ws(t))
                assert b.startswith(t[:len(t) - len(t.lstrip())])


def test_default_is_unchanged_behaviour():
    assert WordTrigger().preserve_whitespace is False
    assert SentenceTrigger().preserve_whitespace is False
    random.seed(1)
    toks = "a  b ".split()
    w = random.choice(["mn"])
    toks.insert(random.randint(0, 2), w)
    random.seed(1)
    assert WordTrigger(words=["mn"]).insert("a  b ") == " ".join(toks)


def test_syntactic_trailing_ws(tmp_path):
    from counterfactual_kd.triggers.syntactic import SyntacticTrigger
    table = tmp_path / "t.jsonl"
    table.write_text(json.dumps({"text": "it is good . ", "para": "when it was , it is good .",
                                 "ok": True}) + "\n")
    off = SyntacticTrigger(table_path=str(table))
    on = SyntacticTrigger(table_path=str(table), preserve_whitespace=True)
    assert off.insert("it is good . ") == "when it was , it is good ."
    assert on.insert("it is good . ") == "when it was , it is good . "


def test_controls_preserving_whitespace_same_draws():
    from counterfactual_kd.data.controls import control_texts
    origs = ["the film  was good . ", "a  plot ", "x y z"]
    random.seed(3)
    trig = [WordTrigger(words=["cf", "mn"], num_triggers=2).insert(t) for t in origs]
    specs = [{"name": "w", "words": ["xv", "zx"]},
             {"name": "ws", "words": ["xv", "zx"], "mode": "substitute"},
             {"name": "s", "sentence": "I saw that old car", "mode": "substitute"},
             {"name": "p", "words": ["soothing sharing how"], "position": "prefix"}]
    for spec in specs:
        state = random.getstate()
        a, ia = control_texts(spec, origs, trig)
        b, ib = control_texts(spec, origs, trig, preserve_whitespace=True)
        assert random.getstate() == state
        assert [norm(x) for x in a] == [norm(x) for x in b]
        assert all(x.endswith(ws.trailing_ws(o)) for x, o in zip(b, origs))
        assert ia == ib


# ── real SST-2 / AG News inputs ──────────────────────────────────────

CONTROLS = [
    {"name": "rare_unplanted", "words": ["xv", "zx", "qk", "vy"]},
    {"name": "rare_unplanted_sub", "words": ["xv", "zx", "qk", "vy"], "mode": "substitute"},
    {"name": "common_neutral", "words": ["table", "window", "paper", "street"]},
    {"name": "loaded_positive", "words": ["great", "wonderful", "excellent", "brilliant"]},
    {"name": "sent_neutral_sub", "sentence": "I saw that old car", "mode": "substitute"},
    {"name": "sent_partial", "sentence": "I watch this"},
    {"name": "atba_phrase_sst2", "words": ["soothing sharing how"], "position": "prefix"},
    {"name": "atba_phrase_agnews", "words": ["infected malicious albums, browser"], "position": "prefix"},
]


def _variants():
    with open(REF) as f:
        ref = json.load(f)["variants"]
    out = []
    for key in ref:
        ds, rest = key.split(":", 1)
        name, *kv = rest.split("|")
        out.append((key, ds, name, dict(x.split("=", 1) for x in kv)))
    return ref, out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def probes():
    """{variant: {"off": probe texts, "on": whitespace-preserving probe texts,
    "rng_off"/"rng_on": RNG state hashes}} as evaluate_model_pair builds them."""
    if os.environ.get("CKD_SKIP_DATA_TESTS"):
        pytest.skip("CKD_SKIP_DATA_TESTS set")
    if not os.path.exists(TABLE):
        pytest.skip("SynBkd table not available")
    from counterfactual_kd.data.controls import control_texts
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import build_attack, dataset_spec
    from counterfactual_kd.triggers.syntactic import SyntacticTrigger

    def no_live(self, text):
        raise RuntimeError(f"SynBkd table miss: {text!r}")

    ref, variants = _variants()
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SyntacticTrigger, "_live", no_live)
        mp.setenv("CKD_SYNBKD_TABLE", TABLE)
        for key, ds, name, ov in variants:
            dspec = dataset_spec(ds)
            attack = build_attack(name, target_label=dspec.target_label, **ov)
            rec = {}
            for mode in ("off", "on"):
                try:
                    test_data = load_test(ds, "sst2_test" if ds == "sst2" else None,
                                          cache_dir=os.environ.get("HF_HOME", None))
                except Exception as e:  # noqa: BLE001
                    pytest.skip(f"test set not available offline: {e}")
                a = ws.preserving_whitespace(attack) if mode == "on" else attack
                _, pl, cl = make_dataloaders(test_data, attack.trigger.trigger_description,
                                             dspec.target_label, None, "prefix", 128, 32, attack=a)
                rng = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()
                trig, clean = list(pl.dataset.texts), list(cl.dataset.texts)
                ctrls = {}
                for spec in CONTROLS:
                    if mode == "on":
                        ctrls[spec["name"]] = control_texts(spec, clean, trig, preserve_whitespace=True)
                    else:
                        ctrls[spec["name"]] = control_texts(spec, clean, trig)
                rec[mode] = {"trig": trig, "clean": clean, "controls": ctrls, "rng": rng,
                             "labels": list(pl.dataset.labels),
                             "rng_after_controls":
                                 hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}
            out[key] = rec
    return ref, out


def test_flag_off_probe_texts_byte_identical_to_pre_a15(probes):
    ref, out = probes
    for key, r in ref.items():
        off = out[key]["off"]
        assert len(off["trig"]) == r["n"] and r["n"] in (912, 5700)
        assert _sha(off["trig"]) == r["trig"], key
        assert _sha(off["clean"]) == r["clean"], key
        assert _sha(off["labels"]) == r["labels"], key
        assert off["rng"] == r["rng_after_trig"], key
        assert off["rng_after_controls"] == r["rng_after_controls"], key
        for name, c in r["controls"].items():
            texts, info = off["controls"][name]
            assert _sha(texts) == c["texts"], (key, name)
            assert info == c["info"], (key, name)


def name_of(key: str) -> str:
    return key.split(":", 1)[1].split("|")[0]


def test_whitespace_preserving_normalises_to_official(probes):
    _, out = probes
    for key, rec in out.items():
        off, on = rec["off"], rec["on"]
        assert on["clean"] == off["clean"]
        assert on["rng"] == off["rng"], key                      # same draws
        assert on["rng_after_controls"] == off["rng_after_controls"]
        assert [norm(t) for t in on["trig"]] == [norm(t) for t in off["trig"]], key
        # Random-position insertions (and SynBkd paraphrases) are normalised
        # already: exact match. Prefix/suffix outputs keep the original
        # string, so only their normalised forms are compared.
        exact = [o == norm(o) for o in off["trig"]]
        assert all(norm(w) == o for w, o, e in zip(on["trig"], off["trig"], exact) if e), key
        if name_of(key) not in ("atba",):
            assert sum(exact) >= len(exact) - 1, key       # one AG News paraphrase has "  "
        for o, w in zip(off["clean"], on["trig"]):
            if o.split():
                assert w.endswith(ws.trailing_ws(o)), (key, o, w)
        for name, (texts, info) in off["controls"].items():
            wtexts, winfo = on["controls"][name]
            assert [norm(t) for t in wtexts] == [norm(t) for t in texts], (key, name)
            assert all(norm(w) == t for w, t in zip(wtexts, texts) if t == norm(t)), (key, name)
            assert winfo == info
            assert all(w.endswith(ws.trailing_ws(o)) for w, o in zip(wtexts, off["clean"])
                       if o.split()), (key, name)
        if key.startswith("sst2:"):
            assert all(t.endswith(" ") for t in on["trig"]), key  # A14 item 10 restored


# ── config, paths, Stage 0 planning ──────────────────────────────────

def test_a15_options_default_off_and_not_serialised():
    from counterfactual_kd.config import A15_OPTIONAL_KEYS, ExperimentConfig, _from_dict
    d = ExperimentConfig(methods=["kd"], pairs=[("bert-large-uncased", "bert-base-uncased")],
                         datasets=["sst2"], attacks=["badnets"], kd_types=["logit"]).to_dict()
    for sec, keys in A15_OPTIONAL_KEYS.items():
        assert not set(keys) & set(d[sec]), sec
    cfg = _from_dict({**d, "stage1": {**d["stage1"], "clean_teacher_epoch": 4,
                                      "record_provenance": True}})
    d2 = cfg.to_dict()
    assert d2["stage1"]["clean_teacher_epoch"] == 4 and d2["stage1"]["record_provenance"] is True


def test_a15_validation():
    from counterfactual_kd.config import ExperimentConfig
    base = dict(methods=["kd"], pairs=[("bert-large-uncased", "bert-base-uncased")],
                datasets=["sst2"], attacks=["badnets", "sos"], kd_types=["logit"])
    cfg = ExperimentConfig(**base)
    cfg.stage0.save_every_epoch = True
    with pytest.raises(ValueError, match="base trainer"):
        cfg.validate()
    cfg = ExperimentConfig(**base)
    cfg.stage1.clean_teacher_epoch = 0
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = ExperimentConfig(**base)
    cfg.stage1.poisoned_student_run = {"badnets": "r"}
    with pytest.raises(ValueError, match="sos"):
        cfg.validate()
    cfg = ExperimentConfig(**base)
    cfg.stage0.build_kinds = ["teacher"]
    with pytest.raises(ValueError):
        cfg.validate()


def test_teacher_pair_paths_default_and_epoch(tmp_path):
    from counterfactual_kd.config import ExperimentConfig
    from counterfactual_kd.pipeline.stage1_kd import teacher_pair_paths
    from counterfactual_kd.registry import teacher_clean_path, teacher_poisoned_path
    cfg = ExperimentConfig(methods=["kd"], pairs=[("gpt2-xl", "gpt2")], datasets=["sst2"],
                           attacks=["atba"], kd_types=["logit"])
    cfg.output.teacher_dir = str(tmp_path / "frozen")
    p, c = teacher_pair_paths(cfg, "atba", "gpt2-xl", "sst2", 42)
    assert p == teacher_poisoned_path(cfg.output.teacher_dir, "atba", "gpt2-xl", "sst2", 42)
    assert c == teacher_clean_path(cfg.output.teacher_dir, "atba", "gpt2-xl", "sst2", 42)
    cfg.stage1.clean_teacher_dir = str(tmp_path / "x15")
    cfg.stage1.clean_teacher_epoch = 5
    p2, c2 = teacher_pair_paths(cfg, "atba", "gpt2-xl", "sst2", 42)
    assert p2 == p and c2 == os.path.join(str(tmp_path / "x15"), "clean",
                                          "ob_gpt2-xl_sst-2_seed42", "epochs", "epoch5")
    cfg.stage1.clean_teacher_epoch = "poisoned"
    os.makedirs(p)
    with open(os.path.join(p, "epochs.json"), "w") as f:
        json.dump({"ob_selected_epoch": 3}, f)
    assert teacher_pair_paths(cfg, "atba", "gpt2-xl", "sst2", 42)[1].endswith("epochs/epoch3")


def test_reused_poisoned_student_is_read_from_the_other_run():
    from counterfactual_kd.config import ExperimentConfig
    from counterfactual_kd.pipeline.stage2_eval import _locate_student_pair
    cfg = ExperimentConfig(methods=["kd"], pairs=[("bert-large-uncased", "bert-base-uncased")],
                           datasets=["sst2"], attacks=["badnets"], kd_types=["logit"], seeds=[43],
                           teacher_seed=42, corpus_seed=42)
    cond = cfg.conditions()[0]
    root = "/runs/new/artifacts/students"
    tag, p, c = _locate_student_pair(cond, root)
    assert p.startswith(root) and c.startswith(root)
    cfg.stage1.poisoned_student_run = "tifs_b_bert_sst2_s43"
    tag2, p2, c2 = _locate_student_pair(cond, root, cfg)
    assert tag2 == tag == "badnets_cf_mn_logit_bert_bert-base-uncased_sst2_seed43_t42_c42_s43"
    assert p2 == f"/runs/tifs_b_bert_sst2_s43/artifacts/students/{tag}/poisoned_student"
    assert c2 == c


def test_stage0_build_kinds():
    from counterfactual_kd.config import ExperimentConfig
    from counterfactual_kd.pipeline.stage0_teacher import Stage0TeacherTraining
    cfg = ExperimentConfig(methods=["kd"], pairs=[("bert-large-uncased", "bert-base-uncased")],
                           datasets=["sst2"], attacks=["badnets"], kd_types=["logit"])
    plan = Stage0TeacherTraining()._plan_teachers(cfg)
    assert {j["kind"] for j in plan} == {"poisoned", "clean"}
    cfg.stage0.build_kinds = ["clean"]
    assert [j["kind"] for j in Stage0TeacherTraining()._plan_teachers(cfg)] == ["clean"]


def test_ob_selection_rule_on_logged_dev_numbers():
    """BERT-large SST-2 clean twin, teacher seed 42 (logs/tifs/tifs_t_bert.log;
    selection_builds.csv): OB selects epoch 3, accuracy alone epoch 5."""
    from counterfactual_kd.adapters.openbackdoor import _accuracy_only_epoch, _ob_selected_epoch
    clean = [0.9243, 0.9289, 0.9243, 0.9197, 0.9300]
    hit = [0.0958, 0.0958, 0.1051, 0.1005, 0.0584]
    recs = [{"epoch": e + 1, "dev": {"dev-clean": {"accuracy": a}, "dev-poison": {"accuracy": h}},
             "dev_score": (a + h) / 2} for e, (a, h) in enumerate(zip(clean, hit))]
    assert _ob_selected_epoch(recs) == 3 and _accuracy_only_epoch(recs) == 5
    tie = [dict(r, dev_score=0.5) for r in recs]
    assert _ob_selected_epoch(tie) == 1


def test_save_every_epoch_wrapper_with_fake_trainer(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from counterfactual_kd.adapters.openbackdoor import (TeacherJob, _save_every_epoch,
                                                         _write_epochs_record)

    class Plm:
        def __init__(self):
            self.w = torch.zeros(3)

        def save_pretrained(self, d):
            os.makedirs(d, exist_ok=True)
            save_file({"w": self.w.clone()}, os.path.join(d, "model.safetensors"))

    class Tok:
        def save_pretrained(self, d):
            open(os.path.join(d, "tokenizer.json"), "w").write("{}")

    class Victim:
        plm, tokenizer = Plm(), Tok()

    victim = Victim()
    scores = [0.6, 0.7, 0.7]

    class Trainer:
        def evaluate(self, model, loader, metrics):
            e = int(model.plm.w[0].item())
            return {"dev-clean": {"accuracy": scores[e]}}, scores[e]

        def train(self, model):
            best = None
            for e in range(3):
                model.plm.w = torch.full((3,), float(e))
                _, s = self.evaluate(model, None, None)
                if best is None or s > best[0]:
                    best = (s, model.plm.w.clone())
            model.plm.w = best[1]

    tr = Trainer()
    save = str(tmp_path / "teacher")
    recs = _save_every_epoch(tr, victim, save)
    tr.train(victim)
    victim.plm.save_pretrained(save)
    _write_epochs_record(save, TeacherJob("clean", "badnets", "m", "sst2", 1), recs, None)
    rec = json.load(open(os.path.join(save, "epochs.json")))
    assert rec["ob_selected_epoch"] == 2 and rec["root_matches_selected_epoch"]
    assert [r["epoch"] for r in rec["epochs"]] == [1, 2, 3]
    assert os.path.exists(os.path.join(save, "epochs", "epoch3", "epoch_meta.json"))


# ── queue v9 ─────────────────────────────────────────────────────────

@pytest.mark.skip(reason="stale: written for the queue before A15/A17/A18 were added; scripts/tifs/queue_plan.py (server job schedule) was extended afterwards")
def test_a15_disabled_means_v8_pool():
    import queue_plan as qp
    assert qp.A15_ENABLED is False and qp.A15_B3_ENABLED is False and qp.A15_POOL_JOBS == []
    ids = [j["id"] for j in qp.POOLS["gpu0"]["jobs"]]
    assert not any(i.startswith("tifs_x15_") for i in ids)
    assert not any(j["id"].startswith("tifs_x15_") for lane in qp.LANES.values()
                   for j in lane.get("jobs", []))
    assert ids[-len(qp.A13_JOBS):] == [j["id"] for j in qp.A13_JOBS]


@pytest.mark.skip(reason="stale: written for the queue before A15/A17/A18 were added; scripts/tifs/queue_plan.py (server job schedule) was extended afterwards")
def test_a15_block_is_consistent():
    import analyze
    import lane
    import queue_plan as qp
    a15 = qp.a15_pool_jobs(b3=True)
    pool = qp.POOLS["gpu0"]["jobs"] + a15
    ids = [j["id"] for j in pool]
    assert len(ids) == len(set(ids))
    outside = {"tifs_synbkd_table", "tifs_t_bert", "tifs_ts_bert", qp.RULE_FROZEN, qp.A15_B3_GO, qp.A15_GO}
    known = set(ids) | outside
    runs = set()
    for j in a15:
        assert j["id"].startswith("tifs_x15_") and "mem_gb" in j
        assert all(d in known for d in j.get("after", [])), j["id"]
        if "cmd" in j:
            assert os.path.exists(os.path.join(ROOT, j["cmd"][1])), j["cmd"]
            if "--run" in j["cmd"]:
                runs.add(j["cmd"][j["cmd"].index("--run") + 1])
        else:
            run = j.get("run", j["id"])
            runs.add(run)
            assert os.path.exists(os.path.join(ROOT, "configs", "tifs", f"{run}.yaml")), run
    new = {r for r in runs if r.startswith("tifs_x15_")}
    assert new and all(analyze.parse_run(r) is None for r in new)
    assert not any(r.startswith(("tifs_rep_bert_atba_t43_s", "tifs_rcc_", "tifs_b_", "tifs_x13_"))
                   for r in new)
    trained = {j["id"] for j in a15 if j.get("stages") == ["stage1"]}
    evals = {j["run"]: j for j in a15 if j["id"].endswith("__eval")}
    assert trained == set(evals)
    assert all(e["after"] == [e["run"], qp.RULE_FROZEN, qp.A15_GO] for e in evals.values())
    builds = [j for j in a15 if j["id"].startswith("tifs_x15_bt_bert_")]
    assert [j["after"] for j in builds] == [[qp.A15_GO]] + [[b["id"], qp.A15_GO] for b in builds[:-1]]
    # Every A15 job waits for the hand-written go marker (A13 done).
    assert all(qp.A15_GO in j["after"] for j in a15)
    # Part order of the training jobs: a, b1, b2, c, d, e, e2 (b3 last).
    order = list(qp.a15_parts(True))
    assert order == ["a", "b1", "b2", "c", "d", "e", "e2", "b3"]
    pos = {j["id"]: i for i, j in enumerate(a15)}
    last = -1
    for part, (_, jobs) in qp.a15_parts(True).items():
        first = min(pos[j["id"]] for j in jobs)
        assert first > last, part
        last = max(pos[j["id"]] for j in jobs)
    # The gate blocks every A15-c student; b3 waits for its marker.
    for j in a15:
        if j["id"].startswith("tifs_x15_c_gpt2_atba_s") and not j["id"].endswith("__eval"):
            assert "tifs_x15_c_gate" in j["after"]
        if j["id"].startswith("tifs_x15_b3_") and not j["id"].endswith("__eval"):
            assert qp.A15_B3_GO in j["after"]
    # Simulate the whole v9 pool: memory and slots hold, every job finishes.
    cap, slots = qp.POOLS["gpu0"]["capacity_gb"], qp.POOLS["gpu0"]["slots"]
    states = {d: "ok" for d in outside} | {"tifs_a10b_go": "ok"}
    running, t = {}, 0
    while True:
        while True:
            job, blocked = lane.pick(pool, states.get, {k: m for k, (m, _) in running.items()},
                                     cap, slots, lambda j: j["mem_gb"])
            assert not blocked
            if job is None:
                break
            states[job["id"]] = "running"
            running[job["id"]] = (job["mem_gb"], t + 1 + len(job["id"]) % 5)
            assert sum(m for m, _ in running.values()) <= cap and len(running) <= slots
        if not running:
            break
        done = min(running, key=lambda k: running[k][1])
        t = running.pop(done)[1]
        states[done] = "ok"
    assert all(states.get(j["id"]) == "ok" for j in a15)


def test_a15_configs_match_generator():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "tifs", "aux", "g8-a15",
                                                     "make_configs.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
