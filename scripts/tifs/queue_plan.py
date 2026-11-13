"""TIFS GPU queue v9: which jobs run, on which GPU, in what order.

v2 (2026-10-21) runs the fixed audit over an attack battery
(docs/tifs_plan.md §0.2) plus student seeds 43/44 of the SST-2 logit
batteries (§0.3). GPU 0 does all training: two lanes (gpu0p1, gpu0p2)
draw from one pool in priority order, each job declaring the GPU memory it
may use, so jobs that cannot share the GPU never overlap (a Qwen3 teacher
build takes 60-66 GB; a BERT-large AG News build 34 GB; students 5-28 GB)
and neither lane idles while the other has work. Training jobs stop after
Stage 1; each run's Stage 2 evaluation is a separate job on GPU 1, which
is ~1.85x slower, sometimes switched off, and therefore kept for
restartable inference only. Teachers are built by Stage-0-only jobs so two
lanes never train the same teacher.

v4 (2026-10-22, docs/tifs_addendum.md A1-A2) adds student seeds 43/44 of
the BERT AG News logit battery and a second ATBA teacher build (teacher
seed 43, student seeds 42-44), and reorders the pending jobs: the A/A
teachers at seed 43 and the ATBA-phrase A/A run first (the replicate
reuses their clean teacher), the RIPPLe seeds before the long tail, and
the A/A contaminated-corpus runs, cut first at the deadline (§0.3), last.

v5 (2026-10-23, docs/tifs_addendum.md A8) adds the hidden-state KD runs
tifs_hid_bert_sst2_s{42,43,44}: seed 42 next, as the A8 pilot (its checks
are fixed in A8), and seeds 43/44 last in the pool order, after the
conditional run. lane.pick starts them once no frozen job is ready, so they
can start while tifs_pc_bert_kl10a1 still waits for its GPU 1 go decision;
it then takes the next free slot. On GPU 1 the A8 evaluations come after
every frozen evaluation and the go decision. At the deadline A8 is cut
first (s44, s43, the s42 evaluation), then the A6 order. It also raises the
Qwen3 student's declared memory from 25 to 28 GB, above the ~26 GiB seen in
its AG News phase (no feasible pairing changes).

v6 (2026-10-23) evaluates on GPU 0 as well (author's decision). From
10/22 11:04 KST another user's jobs held GPU 1 back to back, leaving gaps of
2-3 minutes, so the exclusive GPU 1 lane started and yielded the first
evaluation eight times without finishing a cell (29 h without one). The
evaluations and the go decision now head the GPU 0 pool's order, so a slot
that frees up runs the ready evaluations before the next training job; the
A8 evaluations come after every frozen job there, before the A8 seeds 43/44.
GPU 1's lane keeps the same jobs and the same order, and lane.claim marks a
job running under the pool's lock, so each job runs on one GPU. All three
GPUs are H100 PCIe; a GPU 0 re-evaluation of tifs_b_bert_sst2_s43 gave
byte-identical predictions and results (docs/tifs_addendum.md A9).

v7 (2026-10-24, docs/tifs_addendum.md A10) adds two post hoc checks, last
in the GPU 0 order with their evaluations ahead of their training: the CE
coefficient check tifs_rcc_bert_{sst2,agnews} (a CE-only student with
L = 0.5*CE, A10-a), and the second ATBA AG News teacher build with its
students tifs_rep_bert_atba_agnews_t43_s{42,43,44} (A10-b). A10-b waits for
the marker tifs_a10b_go, written by hand from the rule fixed in A10 (.ok to
run, .failed to block). analyze.py does not parse either run name. GPU 1's
lane is unchanged.

v8 (2026-10-26, docs/tifs_addendum.md A13) adds the post hoc bundle A
after every earlier job, evaluations ahead of training as before: Qwen3
AG News student seeds 43/44 (G1), clean reference teachers at teacher
seeds 45/46 with their students (G2), a near-delta positive control with
0.01-0.3% label-kept BadNets rows (G3) and an unsaturated contaminated
corpus with 0.1-1% label-flipped rows (G4). Run ids are tifs_x13_*, which
analyze.py does not parse. GPU 1's lane is unchanged.

v9 (2026-10-26, docs/tifs_addendum.md A15 and its 10/26 19:55 e2
amendment) adds the post hoc A15 re-experiments after every A13 job, on
the GPU 0 pool only, behind A15_ENABLED (False until the implementation is
verified; while False the pool and the lanes are v8 exactly): (a) official
and whitespace-preserving re-scoring of the decoder students and a BERT
control battery (cmd jobs, scripts/tifs/aux/g8-a15/ws_score.py); (b1) the
seed-42 clean twins rebuilt with every epoch saved under
results/tifs/teachers_x15, CPU checks against the frozen twins, and C
students distilled at each P teacher's saved epoch (the frozen P students
are reused through stage1.poisoned_student_run, never retrained); (b2) the
BERT SST-2 reference twins t43-t46 the same way, C' at epochs 1/4/5 and
A2's C'(t43) at epoch 5; (c) a GPT-2-xl SST-2 ATBA teacher (suffix) whose
gate job (exit 3 = failed) blocks its students; (d) BERT AG News student
seeds 45/46; (e) alpha 0.8/1.0 on BERT AG News; (e2) alpha 1.0 on GPT-2
SST-2. BERT twin builds are chained, so two never run at once. The
conditional b3 (A13 G3 C students at epoch 4) is a second flag,
A15_B3_ENABLED, and also waits for the marker tifs_x15_b3_go. The
registered cut order at the deadline is e2, then e, then d. GPU 1's lane
is unchanged.

v10 (2026-10-30, docs/tifs_addendum.md A17) adds the post hoc GPT-2
coverage after every A15 job, on the GPU 0 pool only, behind A17_ENABLED
and the marker tifs_x17_go: the GPT-2-xl AG News clean twin (its 10/21
build was stopped after epoch 1) and a GPT-2-xl SST-2 SynBkd teacher, both
with every epoch saved under results/tifs/teachers_x17; their gates (exit 3
= failed, which blocks the part's students); the AG News BadNets / AddSent
students (P from the 10/21 poisoned teachers, C from the twin's epoch 5) and
the SynBkd students (C from the A15-b1 twin at the P epoch), seeds 42-44;
two conditional C sets read by the frozen procedure (twin's OB-selected
checkpoint), each run only when a CPU decision finds its epoch differs; and
whitespace-preserving re-scores of every A17 run. The AG News build is
declared at the pool's capacity (it runs beside CPU jobs only). GPU 1's
lane is unchanged.

v11 (2026-10-31, docs/tifs_addendum.md A18) adds GPT-2 AG News ATBA after
every A17 job, on the GPU 0 pool only, behind A18_ENABLED and the marker
tifs_x18_go: a GPT-2-xl AG News ATBA teacher (registry GPT-2 AG News trigger,
suffix) with every epoch saved under results/tifs/teachers_x17 next to the
A17 twin, A15-c's gate script on that pair (exit 3 = failed, which blocks
the students), P and C students at seeds 42-44 (C from the twin at the P
epoch), their Stage 2 on GPU 0 and whitespace-preserving re-scores. The
build is declared at the pool's capacity as the A17 twin was. GPU 1's lane
is unchanged.

The static lanes gpu0a/gpu0b of v2 keep only the job they are running
when this plan is installed, then exit; the pool lists those jobs too, so
their memory is counted.
"""

import json
import os

OB_PATH = "/data/Youngho/claude_experiment/KD_defense/code/_external/openbackdoor"
STATUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                      "runs", "tifs", "_status")

# GPU memory a job may use (GB), measured on GPU 0 (80 GB) on 10/20-21:
# Qwen3-1.7B teacher builds 63-66 on SST-2 and AG News alike (padded to
# max_len 512), BERT-large AG News builds 34, RIPPLe SST-2 15, BERT
# students 4. A Qwen3-0.6B student peaked at 17,686 MiB (17.3 GiB) on
# SST-2 and about 26 GiB on AG News (nvidia-smi, 10/22); GPT-2 students are
# not measured yet (upper estimate, fp32 student + Adam + bf16 teacher).
Q_BUILD, B_BUILD, R_BUILD = 66, 34, 16
QWEN3_KD, GPT2_KD, BERT_KD = 28, 10, 6

SYNBKD_TABLE = {"id": "tifs_synbkd_table",
                "cmd": ["python", "scripts/tifs/build_synbkd_table.py",
                        "--datasets", "sst2", "agnews", "--wait", "--workers", "8"]}


def job(rid, mem, after=(), stages=("stage1",), run=None):
    j = {"id": rid, "mem_gb": mem, "after": list(after), "stages": list(stages)}
    if run:
        j["run"] = run
    return j


QT = ["tifs_t_qwen3"]                       # Qwen3 seed-42 teachers
QTS = ["tifs_t_qwen3", "tifs_ts_qwen3"]     # ... and the SynBkd teacher
BT = ["tifs_t_bert", "tifs_ts_bert"]
BTA = BT + ["tifs_tb_bert"]                 # ... and the AG News ATBA teacher
REP_T = ["tifs_rep_bert_atba_t43_teachers"]

# Priority order (v4). Done or running when v4 was installed: the first
# nine. Then the remaining teacher builds; the ATBA reference and
# replicate runs (the only clean-corpus positive rests on one teacher
# build); the first Qwen3 student, so its memory and speed are measured
# early; the BERT AG News battery; the RIPPLe seeds (a missing seed would
# shrink the BERT SST-2 pool, analyze.py pool_seed_variance); the AG News
# seeds; the Qwen3 and GPT-2 batteries; the A/A runs at teacher seed 44;
# the dose point; and the A/A contaminated-corpus runs. Order chosen by a
# schedule simulation (lane.pick with the 10/21 speeds).
GPU0_JOBS = [
    job("tifs_t_qwen3", Q_BUILD, stages=["stage0"]),
    job("tifs_b_bert_sst2", BERT_KD, BT),
    job("tifs_ts_qwen3", Q_BUILD, ["tifs_synbkd_table"], ["stage0"]),
    job("tifs_aa_qwen3_t43_teachers", Q_BUILD, stages=["stage0"],
        run="tifs_aa_qwen3_clean_t43"),
    job("tifs_b_bert_sst2_s43", BERT_KD, BT),
    job("tifs_b_bert_sst2_s44", BERT_KD, BT),
    job("tifs_g_bert_badnets10_s42", BERT_KD, ["tifs_t_bert"]),
    job("tifs_g_bert_addsent10_s42", BERT_KD, ["tifs_t_bert"]),
    job("tifs_pc_bert_kl10", BERT_KD, ["tifs_t_bert"]),
    job("tifs_tb_bert", B_BUILD, stages=["stage0"]),
    job("tifs_aa_bert_t43_teachers", B_BUILD, stages=["stage0"],
        run="tifs_aa_bert_clean_t43"),
    job("tifs_tr_bert_sst2", R_BUILD, stages=["stage0"]),
    # The replicate's clean teacher is the A/A one (same path, chosen on the
    # BadNets dev split like the seed-42 clean teacher), so it waits for it.
    job("tifs_rep_bert_atba_t43_teachers", B_BUILD, ["tifs_aa_bert_t43_teachers"],
        ["stage0"], run="tifs_rep_bert_atba_t43_s42"),
    job("tifs_aa_bert_clean_t43", BERT_KD, ["tifs_aa_bert_t43_teachers"]),
    job("tifs_rep_bert_atba_t43_s42", BERT_KD, REP_T),
    job("tifs_rep_bert_atba_t43_s43", BERT_KD, REP_T),
    job("tifs_rep_bert_atba_t43_s44", BERT_KD, REP_T),
    job("tifs_g_qwen3_badnets10_s42", QWEN3_KD, QT),
    job("tifs_b_bert_agnews", BERT_KD, BTA),
    # v5 (A8): the hidden-state KD pilot, seed 42, next.
    job("tifs_hid_bert_sst2_s42", BERT_KD, BTA),
    job("tifs_br_bert_sst2", BERT_KD, ["tifs_t_bert", "tifs_tr_bert_sst2"]),
    job("tifs_br_bert_sst2_s44", BERT_KD, ["tifs_t_bert", "tifs_tr_bert_sst2"]),
    job("tifs_br_bert_sst2_s43", BERT_KD, ["tifs_t_bert", "tifs_tr_bert_sst2"]),
    job("tifs_b_bert_agnews_s43", BERT_KD, BTA),
    job("tifs_b_bert_agnews_s44", BERT_KD, BTA),
    job("tifs_b_qwen3_sst2_s44", QWEN3_KD, QTS),
    job("tifs_b_qwen3_agnews", QWEN3_KD, QT),
    job("tifs_b_qwen3_sst2", QWEN3_KD, QTS),
    job("tifs_b_qwen3_sst2_s43", QWEN3_KD, QTS),
    job("tifs_aa_qwen3_clean_t43", QWEN3_KD, ["tifs_aa_qwen3_t43_teachers"]),
    job("tifs_b_gpt2_sst2", GPT2_KD),
    job("tifs_b_gpt2_sst2_s44", GPT2_KD),
    job("tifs_b_gpt2_sst2_s43", GPT2_KD),
    job("tifs_aa_bert_t44_teachers", B_BUILD, stages=["stage0"],
        run="tifs_aa_bert_clean_t44"),
    job("tifs_aa_bert_clean_t44", BERT_KD, ["tifs_aa_bert_t44_teachers"]),
    job("tifs_pc_bert_kl01", BERT_KD, ["tifs_t_bert"]),
    job("tifs_aa_qwen3_badnets10_t43", QWEN3_KD, ["tifs_aa_qwen3_t43_teachers"]),
    job("tifs_aa_bert_badnets10_t43", BERT_KD, ["tifs_aa_bert_t43_teachers"]),
    job("tifs_aa_bert_badnets10_t44", BERT_KD, ["tifs_aa_bert_t44_teachers"]),
    # Conditional (§0.3): runs only if PC_ALPHA1_GO succeeds, i.e. the
    # positive control failed; otherwise it is blocked.
    job("tifs_pc_bert_kl10a1", BERT_KD, ["tifs_t_bert", "tifs_pc_kl10a1_go"]),
    # v5 (A8): hidden-state KD seeds 43/44, last in the pool order.
    job("tifs_hid_bert_sst2_s43", BERT_KD, BTA),
    job("tifs_hid_bert_sst2_s44", BERT_KD, BTA),
]

# v7 (A10): post hoc checks, after every frozen and A8 job. A10-b's teacher
# build waits for the hand-written go marker (A10 rule on the A3.4 AG News
# null); its clean teacher is the A/A one at teacher seed 43.
A10B_GO = "tifs_a10b_go"
A10B_T = ["tifs_rep_bert_atba_agnews_t43_teachers"]
A10_JOBS = [
    job("tifs_rcc_bert_sst2", BERT_KD, BT),
    job("tifs_rcc_bert_agnews", BERT_KD, BTA),
    job("tifs_rep_bert_atba_agnews_t43_teachers", B_BUILD, [A10B_GO], ["stage0"],
        run="tifs_rep_bert_atba_agnews_t43_s42"),
    job("tifs_rep_bert_atba_agnews_t43_s42", BERT_KD, A10B_T),
    job("tifs_rep_bert_atba_agnews_t43_s43", BERT_KD, A10B_T),
    job("tifs_rep_bert_atba_agnews_t43_s44", BERT_KD, A10B_T),
]

# Training runs whose Stage 2 is a separate job, in evaluation priority
# order; the A8 runs (added after registration) after every frozen
# evaluation and the frozen go decision.
A8_RUNS = ["tifs_hid_bert_sst2_s42", "tifs_hid_bert_sst2_s43", "tifs_hid_bert_sst2_s44"]
EVAL_RUNS = [j["id"] for j in GPU0_JOBS
             if j["stages"] == ["stage1"] and not j["id"].endswith("_teachers")
             and j["id"] not in A8_RUNS] + A8_RUNS

# Stage 2 waits for the analysis rule to be frozen (docs/tifs_plan.md
# §0.3): runs/tifs/_status/tifs_rule_frozen.ok is written once the rule,
# configs and analysis code are snapshotted, so no v2 prediction exists
# before the rule that reads it.
RULE_FROZEN = "tifs_rule_frozen"


# Pre-registered decision on GPU 1's queue (CPU only): exit 0 = the
# positive control failed, run tifs_pc_bert_kl10a1; exit 3 = not needed
# (the job is marked failed, which blocks the conditional run).
PC_ALPHA1_GO = {"id": "tifs_pc_kl10a1_go",
                "cmd": ["python", "scripts/tifs/decide_conditionals.py", "pc_alpha1"],
                "after": ["tifs_pc_bert_kl10__eval"]}


def evaluate(runs):
    return [{"id": f"{rid}__eval", "run": rid, "after": [rid, RULE_FROZEN],
             "stages": ["stage2"]}
            for rid in runs]


FROZEN_EVALS = evaluate([r for r in EVAL_RUNS if r not in A8_RUNS])
A8_EVALS = evaluate(A8_RUNS)
A10_EVALS = evaluate([j["id"] for j in A10_JOBS if j["stages"] == ["stage1"]])


# v8 (A13): post hoc bundle A. G2's teachers (BERT-large, both datasets,
# teacher seeds 45/46) are Stage-0-only jobs; G1 next to them; then the
# BERT students of G2, G3 and G4 (seed-42 teachers).
A13_G3 = [f"tifs_x13_pcb_{r}_s{s}" for r in ("r0001", "r0003", "r0010", "r0030")
          for s in (42, 43, 44)]
A13_G4 = [f"tifs_x13_gb_{r}_s{s}" for r in ("r0010", "r0030", "r0100") for s in (42, 43, 44)]
A13_JOBS = [
    job("tifs_x13_bert_ref_t45_teachers", B_BUILD, stages=["stage0"], run="tifs_x13_bert_ref_t45"),
    job("tifs_x13_qwen3_agnews_s43", QWEN3_KD, QT),
    job("tifs_x13_qwen3_agnews_s44", QWEN3_KD, QT),
    job("tifs_x13_bert_ref_t46_teachers", B_BUILD, stages=["stage0"], run="tifs_x13_bert_ref_t46"),
    job("tifs_x13_bert_ref_t45", BERT_KD, ["tifs_x13_bert_ref_t45_teachers"]),
    job("tifs_x13_bert_ref_t46", BERT_KD, ["tifs_x13_bert_ref_t46_teachers"]),
] + [job(r, BERT_KD, ["tifs_t_bert"]) for r in A13_G3 + A13_G4]
A13_EVALS = evaluate([j["id"] for j in A13_JOBS if j["stages"] == ["stage1"]])

# v6: GPU memory a Stage 2 evaluation may use on GPU 0 (GB). Upper
# estimates: the BERT ones are measured below 2 GiB, the others are not
# measured; with two slots they never bind next to a training job.
EVAL_MEM = {"bert": 8, "gpt2": 10, "qwen3": 20}


def on_gpu0(evals):
    return [dict(e, mem_gb=EVAL_MEM[e["run"].split("_")[2]]) for e in evals]


# A13 run ids carry the family in their third or fourth field
# (tifs_x13_qwen3_..., tifs_x13_bert_ref_..., tifs_x13_pcb_... = BERT).
A13_EVALS_GPU0 = [dict(e, mem_gb=EVAL_MEM["qwen3" if "_qwen3_" in e["run"] else "bert"])
                  for e in A13_EVALS]

# v9 (A15, docs/tifs_addendum.md A15 and its 10/26 19:55 e2 amendment;
# go marker and ws --teachers added 10/26 after independent verification):
# post hoc re-experiments after every A13 job, GPU 0 pool only. Nothing of
# it is in the pool until A15_ENABLED is set (after verification); while it
# is False the pool and the lanes are exactly v8. Part order a -> b1 -> b2 ->
# c -> d -> e -> e2, evaluations first as before. Teachers go to the new
# root results/tifs/teachers_x15 and runs are tifs_x15_*, which analyze.py
# does not parse; the outputs are described in scripts/tifs/aux/g8-a15/.
A15_ENABLED = True
# A15-b3 (conditional on b1, fixed before data): A13 G3 C students at epoch
# 4. Even when enabled the jobs wait for the hand-written marker
# tifs_x15_b3_go (.ok to run, .failed to block).
A15_B3_ENABLED = False
A15_B3_GO = "tifs_x15_b3_go"
# Every A15 job also waits for the hand-written marker tifs_x15_go (.ok once
# every A13 job is terminal; verifier advice 10/26), so A15 cannot start in
# the A13 tail and the lanes do not exit between the two bundles.
A15_GO = "tifs_x15_go"

# GPT-2-xl teacher build: not measured on SST-2 (its AG News builds ran out
# of the 80 GB); declared above the Qwen3 build, so it pairs only with a
# BERT student or a CPU check.
G_BUILD = 72
X15_DIR = "scripts/tifs/aux/g8-a15"
X15_TEACHERS = "results/tifs/teachers_x15"
FROZEN_TEACHERS = "results/tifs/teachers"


def _fam(run_id):
    return "qwen3" if "_qwen3_" in run_id else "gpt2" if "_gpt2_" in run_id else "bert"


def _a15_eval(runs):
    return [dict(e, mem_gb=EVAL_MEM[_fam(e["run"])]) for e in evaluate(runs)]


def _a15_ws(run_id):
    """A15-a: official and whitespace-preserving re-scoring of a run's students."""
    return {"id": f"tifs_x15_ws__{run_id}", "mem_gb": EVAL_MEM[_fam(run_id)],
            "cmd": ["python", f"{X15_DIR}/ws_score.py", "--run", run_id, "--teachers"],
            "after": [f"{run_id}__eval"]}


def _a15_check(build_id, teacher, log, expect=None, after=()):
    """A15-b: CPU comparison of a rebuilt clean twin with the frozen one."""
    cmd = ["python", f"{X15_DIR}/check_twin.py",
           "--rebuilt", f"{X15_TEACHERS}/clean/{teacher}",
           "--frozen", f"{FROZEN_TEACHERS}/clean/{teacher}",
           "--frozen-log", log,
           "--out", f"runs/tifs/tifs_x15_twincheck/{teacher}.json"]
    if expect is not None:
        cmd += ["--expect-epoch", str(expect)]
    return {"id": f"tifs_x15_twincheck_{teacher}", "mem_gb": 0, "cmd": cmd,
            "after": [build_id, *after]}


def a15_parts(b3=False):
    """{part: (evaluations, jobs)} of A15 in queue order (pure; the pool
    takes them only when A15_ENABLED)."""
    KD = {"bert": BERT_KD, "gpt2": GPT2_KD, "qwen3": QWEN3_KD}
    bt = {("bert", t): f"tifs_x15_bt_bert_sst2_t{t}" for t in (42, 43, 44, 45, 46)}
    bt[("gpt2", 42)] = "tifs_x15_bt_gpt2_sst2_t42"
    bt[("qwen3", 42)] = "tifs_x15_bt_qwen3_sst2_t42"
    ct = "tifs_x15_ct_gpt2_atba_sst2_t42"
    parts = {}

    # a: existing runs (the A13 G1 seeds wait for their evaluations).
    ws_runs = ["tifs_b_gpt2_sst2", "tifs_b_gpt2_sst2_s43", "tifs_b_gpt2_sst2_s44",
               "tifs_b_qwen3_sst2", "tifs_b_qwen3_sst2_s43", "tifs_b_qwen3_sst2_s44",
               "tifs_b_qwen3_agnews", "tifs_x13_qwen3_agnews_s43", "tifs_x13_qwen3_agnews_s44",
               "tifs_g_qwen3_badnets10_s42", "tifs_aa_qwen3_badnets10_t43",
               "tifs_b_bert_sst2"]
    parts["a"] = ([], [_a15_ws(r) for r in ws_runs])

    # b1: the seed-42 twins (BERT first; BERT twin builds are chained, so
    # two never run at once), their checks, then the epoch-matched C students.
    b1_builds = [job(bt[("bert", 42)], B_BUILD, stages=["stage0"]),
                 job(bt[("gpt2", 42)], G_BUILD, stages=["stage0"]),
                 job(bt[("qwen3", 42)], Q_BUILD, stages=["stage0"])]
    b1_checks = [
        _a15_check(bt[("bert", 42)], "ob_bert-large_sst-2_seed42", "logs/tifs/tifs_t_bert.log", 3),
        _a15_check(bt[("gpt2", 42)], "ob_gpt2-xl_sst-2_seed42", "logs/tifs/tifs_t_gpt2.log", 2),
        _a15_check(bt[("qwen3", 42)], "ob_Qwen3-1.7B_sst-2_seed42", "logs/tifs/tifs_t_qwen3.log", 1)]
    b1_runs = ([(f"tifs_x15_b1_bert_sst2_e{e}_s{s}", "bert") for e in (1, 4, 5) for s in (42, 43, 44)]
               + [(f"tifs_x15_b1_gpt2_sst2_e5_s{s}", "gpt2") for s in (42, 43, 44)]
               + [(f"tifs_x15_b1_qwen3_sst2_e3_s{s}", "qwen3") for s in (42, 43, 44)])
    b1_students = [job(r, KD[f], [bt[(f, 42)]]) for r, f in b1_runs]
    parts["b1"] = (_a15_eval([r for r, _ in b1_runs]), b1_builds + b1_checks + b1_students)

    # b2: reference twins t43-t46 (chained after the t42 build), checks, C'
    # students at epochs 1/4/5 (t45/t46 reuse A13 G2's BadNets P students), A2.
    b2_builds = [job(bt[("bert", t)], B_BUILD, [bt[("bert", t - 1)]], ["stage0"])
                 for t in (43, 44, 45, 46)]
    b2_checks = [
        _a15_check(bt[("bert", 43)], "ob_bert-large_sst-2_seed43",
                   "logs/tifs/tifs_aa_bert_t43_teachers.log", 1),
        _a15_check(bt[("bert", 44)], "ob_bert-large_sst-2_seed44",
                   "logs/tifs/tifs_aa_bert_t44_teachers.log", 3),
        _a15_check(bt[("bert", 45)], "ob_bert-large_sst-2_seed45",
                   "logs/tifs/tifs_x13_bert_ref_t45_teachers.log", 5),
        _a15_check(bt[("bert", 46)], "ob_bert-large_sst-2_seed46",
                   "logs/tifs/tifs_x13_bert_ref_t46_teachers.log",
                   after=["tifs_x13_bert_ref_t46_teachers"])]
    p_runs = {43: [], 44: [], 45: ["tifs_x13_bert_ref_t45"], 46: ["tifs_x13_bert_ref_t46"]}
    b2_runs = [(f"tifs_x15_b2_bert_ref_t{t}_e{e}", [bt[("bert", t)], *p_runs[t]])
               for t in (43, 44, 45, 46) for e in (1, 4, 5) if (t, e) != (43, 1)]
    b2_runs += [(f"tifs_x15_b2_a2_t43_e5_s{s}", [bt[("bert", 43)]]) for s in (43, 44)]
    b2_students = [job(r, BERT_KD, deps) for r, deps in b2_runs]
    parts["b2"] = (_a15_eval([r for r, _ in b2_runs]), b2_builds + b2_checks + b2_students)

    # c: GPT-2 ATBA teacher (after the GPT-2 twin build), its gate (exit 3
    # = failed, which blocks the students), P and C students.
    c_runs = [f"tifs_x15_c_gpt2_atba_s{s}" for s in (42, 43, 44)]
    c_jobs = ([job(ct, G_BUILD, [bt[("gpt2", 42)]], ["stage0"]),
               {"id": "tifs_x15_c_gate", "mem_gb": EVAL_MEM["gpt2"],
                "cmd": ["python", f"{X15_DIR}/gate_c.py"], "after": [ct, bt[("gpt2", 42)]]}]
              + [job(r, GPT2_KD, [ct, "tifs_x15_c_gate", bt[("gpt2", 42)]]) for r in c_runs]
              + [_a15_ws(r) for r in c_runs])
    parts["c"] = (_a15_eval(c_runs), c_jobs)

    # d: BERT AG News battery at student seeds 45/46.
    d_runs = [f"tifs_x15_d_bert_agnews_s{s}" for s in (45, 46)]
    parts["d"] = (_a15_eval(d_runs), [job(r, BERT_KD, BTA) for r in d_runs])

    # e: alpha 0.8 / 1.0, BERT AG News.
    e_runs = [f"tifs_x15_e_bert_agnews_{a}_s{s}" for a in ("a08", "a10") for s in (42, 43, 44)]
    parts["e"] = (_a15_eval(e_runs), [job(r, BERT_KD, BTA) for r in e_runs])

    # e2 (amendment): alpha 1.0, GPT-2 SST-2, C from the b1 twin at each P epoch.
    e2_runs = [f"tifs_x15_e2_gpt2_sst2_a10_e{e}_s{s}" for e in (2, 5) for s in (42, 43, 44)]
    parts["e2"] = (_a15_eval(e2_runs), [job(r, GPT2_KD, [bt[("gpt2", 42)]]) for r in e2_runs])

    if b3:
        b3_runs = [f"tifs_x15_b3_pcb_{r}_s{s}" for r in ("r0001", "r0003", "r0010", "r0030")
                   for s in (42, 43, 44)]
        parts["b3"] = (_a15_eval(b3_runs),
                       [job(r, BERT_KD, [bt[("bert", 42)], "tifs_x13_pcb_" + r[len("tifs_x15_b3_pcb_"):],
                                         A15_B3_GO]) for r in b3_runs])
    return parts


def a15_pool_jobs(b3=False):
    """The A15 block of the GPU 0 pool: every evaluation (part order), then
    the jobs of a, b1, b2, c, d, e, e2 (and b3)."""
    parts = a15_parts(b3)
    jobs = ([e for ev, _ in parts.values() for e in ev]
            + [j for _, jobs in parts.values() for j in jobs])
    return [dict(j, after=[*j.get("after", []), A15_GO]) for j in jobs]


A15_POOL_JOBS = a15_pool_jobs(A15_B3_ENABLED) if A15_ENABLED else []

# v10 (A17, docs/tifs_addendum.md A17, author decision 10/30): post hoc GPT-2 coverage after
# every A15 job, GPU 0 pool only, behind A17_ENABLED and the hand-written marker tifs_x17_go.
# Teachers go to results/tifs/teachers_x17 and runs are tifs_x17_*, which analyze.py does not
# parse; the outputs are described in scripts/tifs/aux/g11-a17/README.md.
A17_ENABLED = True
A17_GO = "tifs_x17_go"
X17_DIR = "scripts/tifs/aux/g11-a17"
# GPT-2-xl AG News build: one process held 79 GB on 10/20 (out of memory, after an SST-2 build
# in the same process); declared at the pool's capacity, so it runs beside CPU jobs only.
G_BUILD_AG = 78


def _a17_ws(run_id):
    """Official and whitespace-preserving re-scoring of an A17 run's students (A15-a script)."""
    return {"id": f"tifs_x17_ws__{run_id}", "mem_gb": EVAL_MEM["gpt2"],
            "cmd": ["python", f"{X15_DIR}/ws_score.py", "--run", run_id, "--teachers",
                    "--out", "runs/tifs/tifs_x17_ws"],
            "after": [f"{run_id}__eval"]}


def _a17_cmd(jid, args, after, mem=0):
    return {"id": jid, "mem_gb": mem, "cmd": ["python", *args], "after": list(after)}


def a17_pool_jobs():
    """The A17 block of the GPU 0 pool: evaluations, builds (AG News twin, then the SynBkd
    teacher), CPU decisions, gates, students, conditional students, ws re-scores."""
    gt, st = "tifs_x17_gt_gpt2_agnews_t42", "tifs_x17_st_gpt2_synbkd_sst2_t42"
    ag = [f"tifs_x17_ag_gpt2_agnews_s{s}" for s in (42, 43, 44)]
    sb = [f"tifs_x17_sb_gpt2_sst2_s{s}" for s in (42, 43, 44)]
    agob = [f"tifs_x17_agob_gpt2_agnews_s{s}" for s in (42, 43, 44)]
    sbob = [f"tifs_x17_sbob_gpt2_sst2_s{s}" for s in (42, 43, 44)]
    runs = ag + sb + agob + sbob
    # The AG News twin build ran out of memory at the start of epoch 2 on 10/30 (attempt 1: 63.5 GB
    # allocated, 15.0 GB reserved but unallocated, after the epoch-1 save of save_every_epoch). It is
    # re-run with the allocator's expandable segments (memory layout only; no numerical effect).
    gt_job = {"id": gt, "mem_gb": G_BUILD_AG, "after": [],
              "cmd": ["env", "PYTORCH_ALLOC_CONF=expandable_segments:True", "python", "-m",
                      "counterfactual_kd", "run", f"configs/tifs/{gt}.yaml", "--run-id", gt,
                      "--resume", "--stages", "stage0"]}
    jobs = ([dict(e, mem_gb=EVAL_MEM["gpt2"]) for e in evaluate(runs)]
            + [gt_job,
               job(st, G_BUILD, [gt], ["stage0"]),
               _a17_cmd("tifs_x17_repro", [f"{X17_DIR}/decide.py", "--what", "repro"], [gt]),
               _a17_cmd("tifs_x17_agob_go", [f"{X17_DIR}/decide.py", "--what", "agob"], [gt]),
               _a17_cmd("tifs_x17_sbob_go", [f"{X17_DIR}/decide.py", "--what", "sbob"], [st]),
               _a17_cmd("tifs_x17_gate_ag", [f"{X17_DIR}/gate.py", "--part", "ag"], [gt],
                        EVAL_MEM["gpt2"]),
               _a17_cmd("tifs_x17_gate_sb", [f"{X17_DIR}/gate.py", "--part", "sb"], [st],
                        EVAL_MEM["gpt2"])]
            + [job(r, GPT2_KD, [gt, "tifs_x17_gate_ag"]) for r in ag]
            + [job(r, GPT2_KD, [st, "tifs_x17_gate_sb"]) for r in sb]
            + [job(r, GPT2_KD, [a, "tifs_x17_agob_go"]) for r, a in zip(agob, ag)]
            + [job(r, GPT2_KD, [b, "tifs_x17_sbob_go"]) for r, b in zip(sbob, sb)]
            + [_a17_ws(r) for r in runs])
    return [dict(j, after=[*j.get("after", []), A17_GO]) for j in jobs]


A17_POOL_JOBS = a17_pool_jobs() if A17_ENABLED else []

# v11 (A18, docs/tifs_addendum.md A18, author decision 10/31): post hoc GPT-2 AG News ATBA after
# every A17 job, GPU 0 pool only, behind A18_ENABLED and the hand-written marker tifs_x18_go. The
# teacher goes to results/tifs/teachers_x17 (the root of its clean twin, the A17 AG News twin);
# runs are tifs_x18_*, which analyze.py does not parse (scripts/tifs/aux/g12-a18/README.md).
A18_ENABLED = True
A18_GO = "tifs_x18_go"
X18_DIR = "scripts/tifs/aux/g12-a18"


def a18_pool_jobs():
    """The A18 block of the GPU 0 pool: evaluations, the ATBA teacher build, its gate (A15-c's
    script on the AG News pair), students, ws re-scores."""
    at = "tifs_x18_at_gpt2_atba_agnews_t42"
    runs = [f"tifs_x18_a_gpt2_agnews_s{s}" for s in (42, 43, 44)]
    # Same allocator setting and declared memory as the A17 AG News twin build (see a17_pool_jobs).
    at_job = {"id": at, "mem_gb": G_BUILD_AG, "after": [],
              "cmd": ["env", "PYTORCH_ALLOC_CONF=expandable_segments:True", "python", "-m",
                      "counterfactual_kd", "run", f"configs/tifs/{at}.yaml", "--run-id", at,
                      "--resume", "--stages", "stage0"]}
    gate = {"id": "tifs_x18_gate", "mem_gb": EVAL_MEM["gpt2"], "after": [at],
            "cmd": ["python", f"{X15_DIR}/gate_c.py", "--teacher-dir", "results/tifs/teachers_x17",
                    "--dataset", "agnews", "--out", "runs/tifs/tifs_x18_gate/gate.json"]}
    ws = [{"id": f"tifs_x18_ws__{r}", "mem_gb": EVAL_MEM["gpt2"],
           "cmd": ["python", f"{X15_DIR}/ws_score.py", "--run", r, "--teachers",
                   "--out", "runs/tifs/tifs_x18_ws"],
           "after": [f"{r}__eval"]} for r in runs]
    jobs = ([dict(e, mem_gb=EVAL_MEM["gpt2"]) for e in evaluate(runs)]
            + [at_job, gate]
            + [job(r, GPT2_KD, [at, "tifs_x18_gate"]) for r in runs]
            + ws)
    return [dict(j, after=[*j.get("after", []), A18_GO]) for j in jobs]


A18_POOL_JOBS = a18_pool_jobs() if A18_ENABLED else []

POOLS = {"gpu0": {"gpu": "0", "capacity_gb": 78, "slots": 2,
                  "jobs": (on_gpu0(FROZEN_EVALS) + [dict(PC_ALPHA1_GO, mem_gb=0)]
                           + [j for j in GPU0_JOBS if j["id"] not in A8_RUNS[1:]]
                           + on_gpu0(A8_EVALS)
                           + [j for j in GPU0_JOBS if j["id"] in A8_RUNS[1:]]
                           + on_gpu0(A10_EVALS) + A10_JOBS
                           + A13_EVALS_GPU0 + A13_JOBS + A15_POOL_JOBS + A17_POOL_JOBS
                           + A18_POOL_JOBS)}}


def _still_running(lane):
    """Jobs whose .running marker names ``lane`` (a v2 static lane)."""
    out = []
    for j in GPU0_JOBS + [SYNBKD_TABLE]:
        try:
            with open(os.path.join(STATUS, f"{j['id']}.running")) as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        if info.get("lane") == lane:
            out.append(j)
    return out


LANES = {
    "gpu0a": {"gpu": "0", "jobs": _still_running("gpu0a")},
    "gpu0b": {"gpu": "0", "jobs": _still_running("gpu0b")},
    "gpu0p1": {"pool": "gpu0"},
    "gpu0p2": {"pool": "gpu0"},
    # GPU 1: evaluation only. Each job waits for its training job; the lane
    # takes whichever is ready first. Other users share GPU 1, so the lane
    # never runs beside another process (author's rule, 2026-10-21; see
    # lane.py "exclusive").
    "gpu1": {
        "gpu": "1",
        "ready_first": True,
        "exclusive": True,
        "jobs": FROZEN_EVALS + [PC_ALPHA1_GO] + A8_EVALS,
    },
}
