"""Re-run the S1b-mode BERT AG News setting (tau 0.03, rho 0.3) with two fresh seeds
(2 x 20000 reps) to see whether the largest |z| in s1c_check_s1b.csv (false TRANSFERS
at delta 0.02) is chance. The first-run value and the archived S1b value are read from
s1c_check_s1b.csv. Post hoc / registered auxiliary (A11.10 validation); changes no verdict.
Writes logs/recheck_s1b.json.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python recheck_s1b.py [--out DIR] [--stock-scipy]
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import common as K  # noqa: E402
import s1c as S  # noqa: E402

SETTING = "BERT AG News n=5700 disc=0.015 m=6 r=3 tau=0.03 rho=0.3 delta=0.02"


def job(seed):
    mdl = S.s1b_model(5700, 0.015, 6, 3, 0.3)
    j = {"rule": "s1b_check", "battery": "BERT AG News", "config": "tau=0.03,rho=0.3", "w": 0.5,
         "reps": 20000, "seed": seed, "src": mdl, "T_src": np.full(6, 0.03), "real_df": float("nan"),
         "real_s2": float("nan"), "E_s2": float("nan"), "n_struct": 0}
    res = S.run_job(j)
    acc = res["acc"]
    return {"seed": seed, "reps": res["reps"],
            "false_transfers_all0@0.02": acc["dec"][0.02]["TRANSFERS"] / res["reps"],
            "no_transfer_all0@0.02": acc["dec"][0.02]["NO TRANSFER"] / res["reps"],
            "false_below": float(acc["below"][0] / res["reps"]),
            "coverage": float(acc["cover"].mean() / res["reps"])}


def main():
    ap = K.add_out_arg(argparse.ArgumentParser())
    ap.add_argument("--stock-scipy", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    chk = K.read_csv(os.path.join(args.out, "s1c_check_s1b.csv"))
    row = [r for r in chk if r.get("setting") == SETTING and r.get("metric") == "p_false_transfers_all_zero"][0]
    first, first_reps = float(row["this_sim"]), int(row["this_reps"])
    ref, ref_reps = float(row["s1b_csv"]), int(row["s1b_reps"])
    with Pool(2, initializer=S._init_worker, initargs=(not args.stock_scipy,)) as p:
        out = p.map(job, [777001, 777002])
    tot = sum(o["reps"] for o in out)
    ft = sum(o["false_transfers_all0@0.02"] * o["reps"] for o in out) / tot
    allp = (ft * tot + first * first_reps) / (tot + first_reps)
    summary = {"setting": SETTING, "metric": "P(false TRANSFERS | all mu = 0)", "runs": out,
               "pooled_new_40000": ft, "s1b_csv": ref, "s1b_reps": ref_reps,
               "z_new_vs_s1b": K.z_two_prop(ft, tot, ref, ref_reps),
               "first_run": first, "first_run_reps": first_reps, "all_60000": allp,
               "z_all_vs_s1b": K.z_two_prop(allp, tot + first_reps, ref, ref_reps)}
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    with open(os.path.join(args.out, "logs", "recheck_s1b.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    K.log_cost(args.out, "recheck_s1b.py", t_start)


if __name__ == "__main__":
    main()
