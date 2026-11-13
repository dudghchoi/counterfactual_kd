"""TIFS GPU pool: job selection (scripts/tifs/lane.py pick) and the plan.

CPU only; nothing is launched.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

import lane  # noqa: E402
import queue_plan as qp  # noqa: E402


def J(i, mem, after=()):
    return {"id": i, "mem_gb": mem, "after": list(after)}


def choose(jobs, states=None, running=None, cap=78, slots=2):
    states = states or {}
    job, blocked = lane.pick(jobs, states.get, running or {}, cap, slots,
                             lambda j: j["mem_gb"])
    return (job["id"] if job else None), blocked


def test_first_fit_in_priority_order():
    jobs = [J("a", 6), J("b", 6)]
    assert choose(jobs) == ("a", [])
    assert choose(jobs, {"a": "running"}, {"a": 6}) == ("b", [])


def test_slots_bound_concurrency():
    assert choose([J("c", 1)], running={"a": 6, "b": 6}) == (None, [])


def test_dependencies():
    jobs = [J("a", 6, ["x"]), J("b", 6, ["y"]), J("c", 6, ["z"])]
    got, blocked = choose(jobs, {"x": "running", "y": "failed", "z": "ok"})
    assert got == "c" and blocked == ["b"]


def test_large_head_is_not_starved():
    # A 66 GB build waits for a 34 GB build to finish. Meanwhile only jobs
    # that leave room for it may start (6 + 66 <= 78; 25 + 66 > 78).
    jobs = [J("qbuild", 66), J("qwen_student", 25), J("bert_student", 6)]
    assert choose(jobs, running={"bbuild": 34}) == ("bert_student", [])
    # Once the big job is gone, the head starts beside the small one.
    assert choose(jobs, {"bert_student": "running"}, {"bert_student": 6}) == ("qbuild", [])


def test_head_is_the_first_ready_job():
    # An unready job does not reserve memory.
    jobs = [J("waits", 66, ["x"]), J("qwen_student", 25)]
    assert choose(jobs, {"x": "running"}, {"bbuild": 34}) == ("qwen_student", [])


def test_memory_classes_encode_the_gpu0_rule():
    cap = qp.POOLS["gpu0"]["capacity_gb"]
    assert qp.Q_BUILD + qp.BERT_KD <= cap and qp.Q_BUILD + qp.GPT2_KD <= cap
    assert qp.Q_BUILD + qp.QWEN3_KD > cap and qp.Q_BUILD + qp.B_BUILD > cap
    assert qp.Q_BUILD + qp.R_BUILD > cap
    assert qp.B_BUILD + qp.QWEN3_KD <= cap and 2 * qp.B_BUILD <= cap


def test_plan_is_consistent():
    ids = [j["id"] for j in qp.GPU0_JOBS]
    assert len(ids) == len(set(ids))
    for j in qp.GPU0_JOBS:
        cfg = os.path.join(ROOT, "configs", "tifs", f"{j.get('run', j['id'])}.yaml")
        assert os.path.exists(cfg), cfg
        assert "mem_gb" in j
    known = set(ids) | {"tifs_synbkd_table", "tifs_t_bert", "tifs_ts_bert", "tifs_pc_kl10a1_go"}
    assert all(d in known for j in qp.GPU0_JOBS for d in j["after"])
    evals = [e for e in qp.LANES["gpu1"]["jobs"] if "cmd" not in e]
    assert evals and all(qp.RULE_FROZEN in e["after"] and e["after"][0] == e["run"]
                         for e in evals)
    assert qp.PC_ALPHA1_GO in qp.LANES["gpu1"]["jobs"]
    trained = {j["id"] for j in qp.GPU0_JOBS if j["stages"] == ["stage1"]}
    assert {e["run"] for e in evals} == trained


def test_a8_runs_come_after_the_frozen_plan():
    """docs/tifs_addendum.md A8: s42 right after the BERT AG News battery,
    s43/s44 last in the pool order; on GPU 1 the A8 evaluations after every
    frozen evaluation and the frozen go decision."""
    a8 = ["tifs_hid_bert_sst2_s42", "tifs_hid_bert_sst2_s43", "tifs_hid_bert_sst2_s44"]
    ids = [j["id"] for j in qp.GPU0_JOBS]
    assert ids.index(a8[0]) == ids.index("tifs_b_bert_agnews") + 1
    assert ids[-2:] == a8[1:]
    gpu1 = [j["id"] for j in qp.LANES["gpu1"]["jobs"]]
    assert gpu1[-3:] == [f"{r}__eval" for r in a8]
    assert gpu1.index(qp.PC_ALPHA1_GO["id"]) < gpu1.index(f"{a8[0]}__eval")


@pytest.mark.skip(reason="stale: written for the queue before A15/A17/A18 were added; scripts/tifs/queue_plan.py (server job schedule) was extended afterwards")
def test_gpu0_pool_runs_the_evaluations_first():
    """v6: every evaluation and the go decision are also GPU 0 pool jobs,
    ahead of training, with the A8 evaluations after every frozen job and
    before the A8 seeds 43/44; GPU 1's list and order are unchanged."""
    gpu1 = qp.LANES["gpu1"]["jobs"]
    pool = qp.POOLS["gpu0"]["jobs"]
    ids = [j["id"] for j in pool]
    assert len(ids) == len(set(ids))
    n = len(qp.FROZEN_EVALS) + 1
    assert [{k: v for k, v in j.items() if k != "mem_gb"} for j in pool[:n]] == gpu1[:n]
    a8 = [f"{r}__eval" for r in qp.A8_RUNS]
    assert ids[-5:] == a8 + qp.A8_RUNS[1:]
    assert ids[n:-5] == [j["id"] for j in qp.GPU0_JOBS if j["id"] not in qp.A8_RUNS[1:]]
    assert [j["id"] for j in gpu1] == ids[:n] + a8
    assert all(j["mem_gb"] <= qp.EVAL_MEM["qwen3"] for j in pool if j["id"].endswith("__eval"))


def test_next_ready_passes_over_jobs_running_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "STATUS", str(tmp_path))
    monkeypatch.setattr(lane, "job_alive", lambda run_id: run_id == "r3")
    (tmp_path / "dep.ok").write_text("{}")
    (tmp_path / "e1.running").write_text(f'{{"gpu": "0", "pid": {os.getpid()}, "run": "r1"}}')
    jobs = [{"id": f"e{i}", "run": f"r{i}", "after": ["dep"]} for i in (1, 3, 2)]
    assert lane.next_ready({"jobs": jobs}, "gpu1")["id"] == "e2"
    (tmp_path / "e2.ok").write_text("{}")
    assert lane.next_ready({"jobs": jobs}, "gpu1") is None


def test_claim_marks_a_job_once(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "STATUS", str(tmp_path))
    monkeypatch.setattr(lane, "job_alive", lambda run_id: False)
    job = {"id": "tifs_b_bert_sst2__eval", "run": "tifs_b_bert_sst2"}
    assert lane.claim(job, "gpu1", "1")
    info = lane.read_marker(job["id"], "running")
    assert info["run"] == "tifs_b_bert_sst2" and info["gpu"] == "1"
    assert (tmp_path / "_pool_gpu0.lock").exists()           # taken under the pool lock
    assert not lane.claim(job, "gpu0p1", "0")                 # this lane is alive
    os.remove(lane.marker(job["id"], "running"))
    monkeypatch.setattr(lane, "job_alive", lambda run_id: True)
    assert not lane.claim(job, "gpu1", "1")                   # its run is being executed


@pytest.mark.skip(reason="stale: written for the queue before A15/A17/A18 were added; scripts/tifs/queue_plan.py (server job schedule) was extended afterwards")
def test_simulated_plan_never_overcommits_gpu0():
    """Run the pool to completion with unit durations: memory and slots hold
    at every step, and every job finishes."""
    pool = qp.POOLS["gpu0"]
    jobs = pool["jobs"]
    states = {"tifs_synbkd_table": "ok", "tifs_t_bert": "ok", "tifs_ts_bert": "ok",
              qp.RULE_FROZEN: "ok"}
    running, t = {}, 0
    dur = {j["id"]: 1 + (len(j["id"]) % 5) for j in jobs}
    while True:
        while True:
            job, blocked = lane.pick(jobs, states.get, {k: m for k, (m, _) in running.items()},
                                     pool["capacity_gb"], pool["slots"], lambda j: j["mem_gb"])
            assert not blocked
            if job is None:
                break
            states[job["id"]] = "running"
            running[job["id"]] = (job["mem_gb"], t + dur[job["id"]])
            assert sum(m for m, _ in running.values()) <= pool["capacity_gb"]
            assert len(running) <= pool["slots"]
        if not running:
            break
        done = min(running, key=lambda k: running[k][1])
        t = running.pop(done)[1]
        states[done] = "ok"
    assert all(states.get(j["id"]) == "ok" for j in jobs)


def test_stale_running_marker_counts_as_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "STATUS", str(tmp_path))
    (tmp_path / "x.running").write_text('{"gpu": "0", "pid": 999999999}')
    monkeypatch.setattr(lane, "job_alive", lambda run_id: False)
    assert lane.live_state("x") is None
    assert lane.running_on("0", lambda j: 6) == {}
    monkeypatch.setattr(lane, "job_alive", lambda run_id: True)
    assert lane.live_state("x") == "running"
    assert lane.running_on("0", lambda j: 6) == {"x": 6}
    assert lane.running_on("1", lambda j: 6) == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_running_marker_names_the_run(tmp_path, monkeypatch):
    # A *_teachers job runs under another run id; liveness must check that
    # run, or a lane restart would start a second 66 GB build beside it.
    monkeypatch.setattr(lane, "STATUS", str(tmp_path))
    (tmp_path / "x_teachers.running").write_text(
        '{"gpu": "0", "pid": 999999999, "run": "x_run"}')
    seen = []
    monkeypatch.setattr(lane, "job_alive", lambda run_id: seen.append(run_id) or run_id == "x_run")
    assert lane.live_state("x_teachers") == "running" and seen == ["x_run"]


def test_set_state_never_leaves_a_job_unmarked(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "STATUS", str(tmp_path))
    lane.set_state("j", "running", lane="l")
    lane.set_state("j", "ok", lane="l")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["events.jsonl", "j.ok"]



def test_pc_fallback_decision(tmp_path, monkeypatch):
    import decide_conditionals as dc
    import analyze
    n = 912

    def cells(p_hits):
        return [{"run": "tifs_pc_bert_kl10", "kd": "logit", "single": False, "dataset": ds,
                 "P": [1] * p_hits + [0] * (n - p_hits), "C": [1] * 50 + [0] * (n - 50)}
                for ds in ("agnews", "sst2")]
    monkeypatch.setattr(analyze, "load_cells", lambda runs: cells(300))
    assert dc.pc_alpha1(str(tmp_path)) == dc.NOT_NEEDED      # control passed
    monkeypatch.setattr(analyze, "load_cells", lambda runs: cells(55))
    assert dc.pc_alpha1(str(tmp_path)) == 0                  # failed: run alpha=1
    monkeypatch.setattr(analyze, "load_cells", lambda runs: [])
    assert dc.pc_alpha1(str(tmp_path)) == 2


def _fake_smi(monkeypatch, used, apps, fail=False):
    """nvidia-smi stand-in: GPU "GPU-1" with ``used`` MiB, compute ``apps``."""
    class R:
        def __init__(self, stdout):
            self.returncode, self.stdout, self.stderr = (1 if fail else 0), stdout, ""

    def run(cmd, **kw):
        if "--query-gpu=uuid,memory.used" in cmd:
            return R(f"GPU-1, {used}\n")
        return R("".join(f"{u}, {p}, {m}\n" for u, p, m in apps))
    monkeypatch.setattr(lane.subprocess, "run", run)


def test_gpu_users(monkeypatch):
    monkeypatch.setattr(lane, "_owner", lambda pid: "other")
    monkeypatch.setattr(lane, "_proc_field", lambda pid, i: "100" if pid == "4242" else "7")
    _fake_smi(monkeypatch, 0, [("GPU-0", 7, 4000)])
    assert lane.gpu_users("1") == []
    _fake_smi(monkeypatch, 20513, [("GPU-1", 4242, 20474), ("GPU-0", 7, 4000)])
    assert lane.gpu_users("1") == ["pid 4242 (other, 20474 MiB)"]
    assert lane.gpu_users("1", own_pgid=100) == []          # the lane's own job
    _fake_smi(monkeypatch, 9000, [])
    assert lane.gpu_users("1") == ["9000 MiB not attributed to a listed process"]
    _fake_smi(monkeypatch, 0, [], fail=True)
    assert lane.gpu_users("1") is None


def test_exclusive_job_yields_the_gpu(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(lane, "LOGS", str(tmp_path))
    seen = []

    def watch(pgid):
        seen.append(pgid)
        return "pid 1 (other, 1 MiB)"
    job = {"id": "x", "cmd": ["python", "-c", "import time; time.sleep(120)"]}
    t0 = time.time()
    with pytest.raises(lane.Yielded):
        lane.run_job(job, "", "", watch=watch, poll=0.1)
    assert len(seen) == 2 and time.time() - t0 < 30       # two strikes, then stop
    with pytest.raises(ProcessLookupError):
        os.kill(seen[0], 0)


def test_exclusive_job_ignores_a_single_blip(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "LOGS", str(tmp_path))
    calls = []

    def blip(pgid):
        calls.append(pgid)
        return "pid 1 (other, 1 MiB)" if len(calls) == 1 else ""
    job = {"id": "y", "cmd": ["python", "-c", "import sys, time; time.sleep(1); sys.exit(3)"]}
    assert lane.run_job(job, "", "", watch=blip, poll=0.1) == 3


def test_gpu1_lane_is_exclusive():
    assert qp.LANES["gpu1"].get("exclusive") is True
    assert all(j.get("cmd") is None or "decide_conditionals" in " ".join(j["cmd"])
               for j in qp.LANES["gpu1"]["jobs"])      # cmd steps there are CPU-only
