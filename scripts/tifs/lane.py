#!/usr/bin/env python3
"""Run one GPU lane of the TIFS queue (scripts/tifs/queue_plan.py).

Each job is ``python -m counterfactual_kd run configs/tifs/<id>.yaml
--run-id <id> --resume [--stages ...]``. Jobs run in order; a job waits
until every dependency has an ``.ok`` marker and is skipped (``.blocked``)
if a dependency failed. Re-running a lane resumes: ``.ok`` jobs are
skipped and the pipeline itself skips finished teachers and students.

The plan is re-read after every job, so jobs appended to queue_plan.py are
picked up without restarting the lane. A job whose run id is still being
executed by another process (e.g. left running by a restarted lane) is
waited for and then re-entered with --resume, which only re-evaluates.

A job may name the run it belongs to (``"run"``, default its id), so a
Stage-2-only evaluation job on GPU 1 can follow the training job of the same
run on GPU 0. A lane waits while its GPU is not visible to nvidia-smi (GPU 1
is sometimes switched off); a job that fails while the GPU is gone is put
back in the queue instead of being marked failed.

A lane may instead draw from a shared pool (``{"pool": name}``, jobs in
``queue_plan.POOLS``): every lane of the pool takes the first job, in the
pool's priority order, whose dependencies are met and whose declared
``mem_gb`` fits next to the jobs already running on that GPU (from any
lane, read from their ``.running`` markers), with at most ``slots`` jobs
at once. The first ready job keeps its place: while it does not fit, only
jobs that leave room for it are started (see ``pick``). Picking happens
under a file lock, so two lanes never take the same job.

A job may be listed both on a lane and in a pool (the Stage 2 evaluations
are on GPU 1's lane and in GPU 0's pool, queue_plan v6). The lane then
passes over jobs another lane is running and marks its job running under
the lock of every pool that lists it (``claim``), the lock the pool's lanes
pick under, so each job runs on one GPU at a time.

A lane marked ``exclusive`` (GPU 1, which other users share) never runs
beside another process. Before each GPU job it waits until nvidia-smi shows
no other compute process and no unattributed memory on its GPU for two
checks in a row. While the job runs it checks every minute, and if another
process shows up twice in a row the job is stopped and put back in the
queue. Stage 2 skips finished cells, so a stopped evaluation repeats only
the cell it was in.

Status markers and per-job logs:
    runs/tifs/_status/<id>.{running,ok,failed,blocked}
    logs/tifs/<id>.log
    runs/tifs/_status/events.jsonl   (one line per start / finish)

    python scripts/tifs/lane.py --lane gpu0a
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATUS = os.path.join(ROOT, "runs", "tifs", "_status")
LOGS = os.path.join(ROOT, "logs", "tifs")

sys.path.insert(0, os.path.dirname(__file__))
import importlib  # noqa: E402

import queue_plan  # noqa: E402


def load_plan():
    importlib.reload(queue_plan)
    return queue_plan.LANES, queue_plan.OB_PATH


def job_alive(job_id: str) -> bool:
    """True if some process is running ``counterfactual_kd run ... --run-id <job_id>``."""
    needle = ["--run-id", job_id]
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if "counterfactual_kd" in argv and any(
                argv[i:i + 2] == needle for i in range(len(argv) - 1)):
            return True
    return False


def gpu_available(gpu: str) -> bool:
    """True if nvidia-smi can see GPU ``gpu``."""
    try:
        r = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=name",
                            "--format=csv,noheader"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def _proc_field(pid, index: int) -> str | None:
    """Field ``index`` of /proc/<pid>/stat after the command name (2 = pgrp)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[index]
    except (OSError, IndexError):
        return None


def _owner(pid) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (OSError, KeyError):
        return "?"


def _mib(text: str) -> float:
    try:
        return float(text)
    except ValueError:      # "[N/A]"
        return 0.0


# GPU memory in use that no listed process accounts for, above which a
# process nvidia-smi cannot name is assumed to hold the GPU.
UNLISTED_MIB = 2048


def gpu_users(gpu: str, own_pgid: int | None = None) -> list[str] | None:
    """Other users of GPU ``gpu``, one description each ([] = free).

    Every compute process on the GPU counts except those in process group
    ``own_pgid`` (the lane's running job), and so does memory in use that
    no listed process accounts for. None if nvidia-smi fails.
    """
    def query(*args):
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise OSError(r.stderr)
        return [[x.strip() for x in line.split(",")]
                for line in r.stdout.splitlines() if line.strip()]

    try:
        rows = query("-i", gpu, "--query-gpu=uuid,memory.used")
        apps = query("--query-compute-apps=gpu_uuid,pid,used_memory")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not rows or len(rows[0]) != 2:
        return None
    uuid, used = rows[0][0], _mib(rows[0][1])
    users, listed = [], 0.0
    for app in apps:
        if len(app) != 3 or app[0] != uuid:
            continue
        pid, mib = app[1], _mib(app[2])
        listed += mib
        if own_pgid is not None and _proc_field(pid, 2) == str(own_pgid):
            continue
        users.append(f"pid {pid} ({_owner(pid)}, {mib:.0f} MiB)")
    if used - listed > UNLISTED_MIB:
        users.append(f"{used - listed:.0f} MiB not attributed to a listed process")
    return users


def marker(job_id: str, kind: str) -> str:
    return os.path.join(STATUS, f"{job_id}.{kind}")


def state(job_id: str) -> str | None:
    for kind in ("ok", "failed", "blocked", "running"):
        if os.path.exists(marker(job_id, kind)):
            return kind
    return None


def set_state(job_id: str, kind: str, **info) -> None:
    # Write the new marker before removing the old ones, so a reader never
    # sees the job without one (``state`` ranks ok/failed/blocked first).
    with open(marker(job_id, kind), "w") as f:
        json.dump(info, f)
    for k in ("ok", "failed", "blocked", "running"):
        if k == kind:
            continue
        try:
            os.remove(marker(job_id, k))
        except FileNotFoundError:
            pass
    note(job_id, kind, **info)


def note(job_id: str, kind: str, **info) -> None:
    """Append an event without changing the job's marker."""
    with open(os.path.join(STATUS, "events.jsonl"), "a") as f:
        f.write(json.dumps({"job": job_id, "state": kind,
                            "time": time.strftime("%F %T"), **info}) + "\n")


def wait_until_free(job_id: str, gpu: str, lane_name: str, poll: float = 60) -> None:
    """Block until ``gpu`` has no other user at two checks ``poll`` seconds
    apart; one 'waiting' event per wait."""
    free, waited = 0, False
    while True:
        users = gpu_users(gpu)
        if users == []:
            free += 1
            if free >= 2:
                if waited:
                    note(job_id, "gpu_free", lane=lane_name, gpu=gpu)
                return
        else:
            free = 0
            if not waited:
                note(job_id, "waiting", lane=lane_name, gpu=gpu,
                     reason="GPU in use by another process",
                     users=users if users is not None else ["nvidia-smi failed"])
                waited = True
        time.sleep(poll)


def wait_for(deps: list[str]) -> bool:
    """Block until all deps are ok; False if any failed or was blocked."""
    while True:
        states = {d: state(d) for d in deps}
        if any(s in ("failed", "blocked") for s in states.values()):
            return False
        if all(s == "ok" for s in states.values()):
            return True
        time.sleep(60)


class Yielded(Exception):
    """The job was stopped because another process started using its GPU."""


def _stop(proc) -> None:
    """Stop the job's whole process group: SIGTERM, then SIGKILL after 60 s."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_job(job: dict, gpu: str, ob_path: str, watch=None, poll: float = 60,
            strikes: int = 2) -> int:
    """Run a pipeline job, or ``job["cmd"]`` (argv, python resolved to the
    lane's interpreter) for helper steps such as linking teachers.

    With ``watch`` (process group id -> reason to stop, or ""), the job runs
    in its own process group and is checked every ``poll`` seconds. After
    ``strikes`` reasons in a row it is stopped and Yielded is raised."""
    job_id = job["id"]
    run_id = job.get("run", job_id)
    if job.get("cmd"):
        cmd = [sys.executable if a == "python" else a for a in job["cmd"]]
    else:
        cmd = [sys.executable, "-m", "counterfactual_kd", "run",
               os.path.join("configs", "tifs", f"{job.get('config', run_id)}.yaml"),
               "--run-id", run_id, "--resume"]
        if job.get("stages"):
            cmd += ["--stages", *job["stages"]]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": ob_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "HF_HOME": env.get("HF_HOME", "/data/Youngho/cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    with open(os.path.join(LOGS, f"{job_id}.log"), "a") as log:
        log.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)} (GPU {gpu})\n")
        log.flush()
        if watch is None:
            return subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        hits = 0
        while True:
            try:
                return proc.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                pass
            reason = watch(proc.pid)
            hits = hits + 1 if reason else 0
            if hits >= strikes:
                _stop(proc)
                log.write(f"\n=== {time.strftime('%F %T')} stopped, GPU {gpu} in use: {reason}\n")
                raise Yielded(reason)


TERMINAL = ("ok", "failed", "blocked")


def pick(jobs, state_of, running, capacity, slots, mem_of):
    """Pool scheduling decision; pure, so it can be tested and simulated.

    Args:
        jobs: the pool's jobs in priority order.
        state_of: job id -> marker state (None = pending; a stale
            ``running`` marker must already map to None).
        running: {job id: GB} of the jobs now running on the pool's GPU.
        capacity, slots: GPU memory budget (GB) and maximum concurrent jobs.
        mem_of: job -> GB it may use.

    Returns:
        (job to start or None, ids whose dependency failed → blocked).

    First fit in priority order, with one reservation: the first job whose
    dependencies are met (the head) keeps its place. While it does not fit,
    a later job is started only if the head would still fit beside it once
    the jobs now running finish, so a large build is not starved by a
    stream of smaller jobs.
    """
    blocked = []
    if len(running) >= slots:
        return None, blocked
    used = sum(running.values())
    head = None
    for job in jobs:
        if state_of(job["id"]) is not None:
            continue
        deps = [state_of(d) for d in job.get("after", [])]
        if any(d in ("failed", "blocked") for d in deps):
            blocked.append(job["id"])
            continue
        if not all(d == "ok" for d in deps):
            continue
        m = mem_of(job)
        if head is None:
            head = m
            if used + m <= capacity:
                return job, blocked
        elif used + m <= capacity and m + head <= capacity:
            return job, blocked
    return None, blocked


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def read_marker(job_id: str, kind: str) -> dict:
    try:
        with open(marker(job_id, kind)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def live_state(job_id: str) -> str | None:
    """Like ``state``, but a ``running`` marker whose lane process and
    pipeline run are both gone counts as pending (it is re-run with
    --resume)."""
    s = state(job_id)
    if s == "running":
        info = read_marker(job_id, "running")
        if not (pid_alive(info.get("pid")) or job_alive(info.get("run", job_id))):
            return None
    return s


def running_on(gpu: str, mem_of_id) -> dict:
    """{job id: GB} of live jobs on ``gpu``, from any lane's markers."""
    out = {}
    for f in os.listdir(STATUS):
        if not f.endswith(".running"):
            continue
        job_id = f[:-len(".running")]
        if str(read_marker(job_id, "running").get("gpu")) != str(gpu):
            continue
        if live_state(job_id) == "running":
            out[job_id] = mem_of_id(job_id)
    return out


def run_pool(lane_name: str, pool_name: str) -> None:
    lock_path = os.path.join(STATUS, f"_pool_{pool_name}.lock")
    while True:
        load_plan()
        pool = queue_plan.POOLS[pool_name]
        gpu, cap = pool["gpu"], float(pool["capacity_gb"])
        mem = {j["id"]: float(j.get("mem_gb", cap)) for j in pool["jobs"]}

        def mem_of_id(job_id):
            return mem.get(job_id, cap)      # unknown jobs: whole GPU

        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if all(state(j["id"]) in TERMINAL for j in pool["jobs"]):
                break
            job, blocked = pick(pool["jobs"], live_state, running_on(gpu, mem_of_id),
                                cap, int(pool["slots"]), lambda j: mem_of_id(j["id"]))
            for job_id in blocked:
                set_state(job_id, "blocked", lane=lane_name)
            if job is not None and job_alive(job.get("run", job["id"])):
                job = None
            if job is not None:
                set_state(job["id"], "running", lane=lane_name, gpu=gpu,
                          pid=os.getpid(), host=socket.gethostname(),
                          run=job.get("run", job["id"]), mem_gb=mem_of_id(job["id"]))
        if job is None:
            time.sleep(60)
            continue
        t0 = time.time()
        code = run_job(job, gpu, queue_plan.OB_PATH)
        set_state(job["id"], "ok" if code == 0 else "failed", lane=lane_name,
                  gpu=gpu, exit=code, hours=round((time.time() - t0) / 3600, 2))
    print(f"lane {lane_name} (pool {pool_name}) finished {time.strftime('%F %T')}")


def next_job(lane: dict) -> dict | None:
    for job in lane["jobs"]:
        if state(job["id"]) not in ("ok", "failed", "blocked"):
            return job
    return None


def next_ready(lane: dict, lane_name: str) -> dict | None:
    """First pending job whose dependencies are all ok (lanes with
    ``ready_first``); jobs with a failed dependency are marked blocked.
    Jobs another lane or process is running are passed over."""
    for job in lane["jobs"]:
        if live_state(job["id"]) is not None:
            continue
        deps = {d: state(d) for d in job.get("after", [])}
        if any(s in ("failed", "blocked") for s in deps.values()):
            set_state(job["id"], "blocked", lane=lane_name)
            continue
        if all(s == "ok" for s in deps.values()) and not job_alive(job.get("run", job["id"])):
            return job
    return None


def claim(job: dict, lane_name: str, gpu: str) -> bool:
    """Mark ``job`` running for this lane, unless another lane took it.

    The check and the marker are made under the lock of every pool that
    also lists the job, so a pool lane cannot pick it in between."""
    job_id = job["id"]
    locks = []
    try:
        for name in sorted(n for n, p in queue_plan.POOLS.items()
                           if any(j["id"] == job_id for j in p["jobs"])):
            f = open(os.path.join(STATUS, f"_pool_{name}.lock"), "w")
            locks.append(f)
            fcntl.flock(f, fcntl.LOCK_EX)
        if live_state(job_id) is not None or job_alive(job.get("run", job_id)):
            return False
        set_state(job_id, "running", lane=lane_name, gpu=gpu, pid=os.getpid(),
                  host=socket.gethostname(), run=job.get("run", job_id))
        return True
    finally:
        for f in locks:
            f.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lane", required=True)
    args = p.parse_args()
    os.makedirs(STATUS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    lanes, _ = load_plan()
    if "pool" in lanes[args.lane]:
        run_pool(args.lane, lanes[args.lane]["pool"])
        return

    while True:
        lanes, ob_path = load_plan()
        lane = lanes[args.lane]
        job = next_job(lane)
        if job is None:
            break
        if lane.get("ready_first"):
            job = next_ready(lane, args.lane)
            if job is None:
                time.sleep(60)
                continue
        job_id = job["id"]
        while job_alive(job.get("run", job_id)):
            time.sleep(60)
        if not wait_for(job.get("after", [])):
            set_state(job_id, "blocked", lane=args.lane)
            continue
        while not gpu_available(lane["gpu"]):
            time.sleep(300)
        # Helper steps (cmd) on the exclusive GPU 1 lane are CPU-only
        # (decide_conditionals), so only pipeline jobs wait for the GPU.
        exclusive = bool(lane.get("exclusive")) and not job.get("cmd")
        if exclusive:
            wait_until_free(job_id, lane["gpu"], args.lane)
        if not claim(job, args.lane, lane["gpu"]):
            time.sleep(10)
            continue
        t0 = time.time()
        watch = None
        if exclusive:
            watch = lambda pgid: "; ".join(gpu_users(lane["gpu"], pgid) or [])  # noqa: E731
        try:
            code = run_job(job, lane["gpu"], ob_path, watch=watch)
        except Yielded as e:
            set_state(job_id, "requeued", reason=f"yielded GPU {lane['gpu']}: {e}",
                      lane=args.lane, gpu=lane["gpu"],
                      hours=round((time.time() - t0) / 3600, 2))
            os.remove(marker(job_id, "requeued"))
            continue
        info = dict(lane=args.lane, gpu=lane["gpu"], exit=code,
                    hours=round((time.time() - t0) / 3600, 2))
        if code != 0 and not gpu_available(lane["gpu"]):
            set_state(job_id, "requeued", reason="gpu not visible", **info)
            os.remove(marker(job_id, "requeued"))
            continue
        set_state(job_id, "ok" if code == 0 else "failed", **info)
    print(f"lane {args.lane} finished {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
