# g12-a18: A18 GPT-2 AG News ATBA (implementation, queue v11)

**Status.** Implementation of docs/tifs_addendum.md **A18** (registered 10/31 KST, snapshot
`a18_registration`). Post hoc, outside the primary family: no frozen verdict, battery decision,
paper-wide Holm, A13, A15 or A17 result changes. No pipeline code changes: the runs use A15's
default-off options and A15's `gate_c.py` and `ws_score.py`.

Trigger: the registry's GPT-2 AG News ATBA trigger, "breeding breeding orb download iverse forest
asive abit" (ATBA paper, arXiv:2408.09878, Table 4, GPT-XL / AG's News; checked against the arXiv
HTML on 10/31), at the suffix as in A15-c.

## Files here

| file | what |
|---|---|
| `make_configs.py` | writes the 4 `configs/tifs/tifs_x18_*.yaml` from frozen GPT-2 templates (A15's merge; A15-c's changes with the dataset set to AG News); `--check` verifies the files on disk; `configs_manifest.json` lists each run's template and changed keys |
| `plan_check.py` | CPU dry run (A17's, run glob `tifs_x18_*`) |
| `analysis/` | the registered analysis (written before any A18 Stage 2 output, run once) |

## Runs (queue v11, GPU 0 pool, 2 slots, 78 GB; every job waits for `tifs_x18_go`)

| run | stage | what |
|---|---|---|
| `tifs_x18_at_gpt2_atba_agnews_t42` | 0 | GPT-2-xl AG News ATBA teacher (suffix), every epoch saved, `results/tifs/teachers_x17` (declared 78 GB: runs beside CPU jobs only) |
| `tifs_x18_gate` | cmd | `g8-a15/gate_c.py --teacher-dir results/tifs/teachers_x17 --dataset agnews` -> `runs/tifs/tifs_x18_gate/gate.json` (exit 3 = failed, blocks the students) |
| `tifs_x18_a_gpt2_agnews_s{42,43,44}` | 1 | P ATBA (OB-selected epoch), C from the A17 twin at the same epoch |
| `<run>__eval` | 2 | Stage 2 on GPU 0 |
| `tifs_x18_ws__<run>` | cmd | `ws_score.py --run <run> --teachers --out runs/tifs/tifs_x18_ws` |
