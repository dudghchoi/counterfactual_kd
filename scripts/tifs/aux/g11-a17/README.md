# g11-a17: A17 GPT-2 coverage (implementation, queue v10)

**Status.** Implementation of docs/tifs_addendum.md **A17** (registered 10/30 19:22 KST, snapshot
`a17_registration_20261030T102211Z`). Post hoc, outside the primary family: no frozen verdict, battery decision,
paper-wide Holm, A13 or A15 result changes. No pipeline code changes: the runs use A15's default-off
options (scripts/tifs/aux/g8-a15/README.md) and A15's `ws_score.py`.

## Files here

| file | what |
|---|---|
| `make_configs.py` | writes the 14 `configs/tifs/tifs_x17_*.yaml` from frozen GPT-2 templates (A15's merge); `--check` verifies the files on disk; `configs_manifest.json` lists each run's template and changed keys |
| `plan_check.py` | CPU dry run (A15's, run glob `tifs_x17_*`): per condition the tag, P (trained or reused), C, and the teacher pair |
| `gate.py` | GPU job: frozen teacher gate (ASR >= 0.90, accuracy drop <= 0.05) of each poisoned teacher vs its twin at the same epoch; `--part ag` (exit 3 only if both attacks fail) / `--part sb` (exit 3 if it fails) |
| `decide.py` | CPU jobs: `--what agob` / `--what sbob` (exit 0 = run the conditional procedure-C students, 3 = not needed); `--what repro` (report only: AG News twin epoch-1 dev accuracy vs the 10/21 stopped build) |
| `diag_sb_format.py` | report only, after the sb gate failed (10/30): clean accuracy of the SynBkd teacher's saved epochs and the twin with and without the trailing space of the GLUE-form test text (and on the OB dev text); changes no gate, decision or run |
| `analysis/` | the registered analysis (written before any A17 Stage 2 output, run once) |

## Runs (queue v10, GPU 0 pool, 2 slots, 78 GB; every job waits for `tifs_x17_go`)

| run | stage | what |
|---|---|---|
| `tifs_x17_gt_gpt2_agnews_t42` | 0 | GPT-2-xl AG News clean twin, every epoch saved, `results/tifs/teachers_x17` (declared 78 GB: runs beside CPU jobs only) |
| `tifs_x17_st_gpt2_synbkd_sst2_t42` | 0 | GPT-2-xl SST-2 SynBkd teacher, every epoch saved, `results/tifs/teachers_x17` |
| `tifs_x17_ag_gpt2_agnews_s{42,43,44}` | 1 | P BadNets / AddSent from the 10/21 teachers (`results/tifs/teachers`, saved epoch 5), shared C from the twin's epoch 5; s42 also R (CE-only) |
| `tifs_x17_sb_gpt2_sst2_s{42,43,44}` | 1 | P SynBkd (OB-selected epoch), C from the A15-b1 twin (`teachers_x15`) at that epoch |
| `tifs_x17_agob_gpt2_agnews_s*` | 1 | conditional (`tifs_x17_agob_go`): C from the twin's OB-selected root, P reused |
| `tifs_x17_sbob_gpt2_sst2_s*` | 1 | conditional (`tifs_x17_sbob_go`): C from the frozen SST-2 twin (epoch 2), P reused |
| `<run>__eval` | 2 | Stage 2 on GPU 0 |
| `tifs_x17_ws__<run>` | cmd | `ws_score.py --run <run> --teachers --out runs/tifs/tifs_x17_ws` |

Outputs of the CPU and gate jobs: `runs/tifs/tifs_x17_{gate_ag,gate_sb}/gate.json`,
`runs/tifs/tifs_x17_{agob,sbob,repro}/decision.json`.
