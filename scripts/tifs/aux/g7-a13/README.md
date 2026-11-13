# A13 post hoc analysis - code (docs/tifs_addendum.md A13: G1-G4 and 'common')

Status: A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Outputs: `results/tifs/addendum/aux/g7-a13/` (its README.md has the findings, the registered sentence per group, deviations, self-checks, data snapshot and definitions). This file is written by make_readme.py.

| script | what |
|---|---|
| common.py | paths, frozen imports, the A13 cells through the frozen analyze.load_cells (RUN_RE swapped for the call), the reproduction gate, floors, CSV writer with the `# definition:` line |
| g1_qwen3_agnews.py | G1: frozen primary_table / primary_battery on the frozen cells + G1 cells (own pool), Welch / min df (A15 E3), A3.1 |
| g2_reference.py | G2: functional check, selected epochs, A3.3 with five references (frozen reference_atba_table), A/A auxiliary; `--with-item7`: item 7 five vs three references |
| g3_near_delta.py | G3: four-rate seed pool, per-rate verdicts, own variance, Holm over rates, anchors, sentences |
| g4_low_dose.py | G4: P1-C1, C1-R1, P1-R1 per rate with three-rate pools, sensitivities, sentences, 10% anchor |
| common_floor_jobs.py | A8 floor for every A13 student; jobs, deadline, cuts (events.jsonl) |
| independent.py | adversarial self-review: the four headline readings without any frozen import |
| selfcheck.py | `--gate` (hashes, official tables reproduced) and the full self-check |
| make_readme.py | both README files |
| run_all.sh | everything, CPU only; also runs the A14 scripts score.py refs and item7_reference_k.py (unchanged) with `--out results/tifs/addendum/aux/g7-a13/item7` |

Run: `scripts/tifs/aux/g7-a13/run_all.sh` (`--no-score` skips the CPU fp32 re-scoring; it is resumable anyway). CPU only, nice -n 10, HF offline. Nothing outside `scripts/tifs/aux/g7-a13/` and `results/tifs/addendum/aux/g7-a13/` is written; frozen files are imported, never edited.

## Key findings

- G1: own-pool battery NO TRANSFER@0.05 (also at 0.02 and 0.10; Welch df same) -> "Qwen3 AG News의 전이 없음은 빌린 분산에 기대지 않는다".
- G2: t45/t46 functional; 5-reference ATBA intervals SST-2 [-0.1540, +0.4237], AG News [+0.0021, +0.3765] (logit), INCONCLUSIVE at every delta -> "reference-dependent 문구를 유지하고 df 4를 근거로 든다". Item 7: new EQUIVALENT-changes with t45/t46: sst2/logit/addsent@0.1, sst2/feature/atba@0.1.
- G3: smallest POSITIVE@0.05 rate 0.3%; k +0.0058/+0.0132/+0.0380/+0.9042 (step between 0.1% and 0.3%) -> none of the three registered sentences applies.
- G4: no amplification (P1-R1 upper < 0.05 at every rate); suppression beyond 0.05 at 0.3%,1% only; R1 0.1378/0.9755/0.9978 -> none of the three registered sentences applies.
- Common: 52/52 jobs ok, none cut; no student below the floor; no exclusion.
- Self-checks 32/32; independent check 39/39.
