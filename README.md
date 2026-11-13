# counterfactual_kd

Matched teacher-replacement evaluation of backdoors in knowledge distillation.

This repository holds the code of the manuscript *Beyond Attack Success Rates in Backdoor
Evaluation for Knowledge Distillation* (prepared for submission to IEEE Transactions on
Information Forensics and Security). The per-input predictions, result tables and protocol
snapshots are archived on Zenodo as version 2.0.0 of
[10.5281/zenodo.21772216](https://doi.org/10.5281/zenodo.21772216) (this DOI resolves to the
latest version).

## What is measured

A high attack success rate (ASR) on a distilled student does not by itself identify the teacher
as its source: the trigger may already fire on a clean teacher, and a contaminated distillation
corpus can produce the backdoor without any teacher. For each condition, three students are
trained on the same distillation corpus, student seed, recipe and evaluation inputs:

- **P**, distilled from a backdoored teacher;
- **C**, distilled from a trainer-matched clean teacher;
- **R**, a teacher-free reference student trained on hard labels (`kd_type: ce`).

On triggered inputs whose true label is not the target, the exposure A(M) is the rate at which
student M predicts the target label, and the teacher-replacement contrast is k = A(P) − A(C)
(`P_trig`, `C_trig` in the files). Verdicts compare paired intervals for k with tolerances
δ ∈ {0.02, 0.05, 0.10}: repeated-seed cells pool the student-seed variance within a battery, and
single-seed cells use a paired Tango score interval with transported variance.

## Repository layout

| Path | Content |
|---|---|
| `counterfactual_kd/` | Library: attacks, triggers, distillation objectives, evaluation, paired statistics (`analysis/tifs.py`, `metrics/paired.py`). |
| `configs/tifs/` | Run configurations of the version-2 experiments (`_dropped/`: runs removed from the plan). |
| `scripts/tifs/` | `analyze.py` (result tables), `make_configs.py`, job scheduling, and the follow-up analyses in `aux/g1`–`g12`. |
| `tests/` | Unit tests. |
| `configs/tier1_*.yaml`, `results/cells/`, `results/dashboard/`, `examples/`, `docs/`, `run.py` | Version-1 configurations, fixtures and documentation (Cohen's h / ROPE dashboard). The v1 verdicts are superseded; see below. |

## Install

Python 3.10 or later.

```bash
pip install -e .                     # library (numpy, scipy)
pip install -e ".[repro]"            # training and evaluation: torch, transformers, datasets, ...
pip install -e ".[teacher]"          # backdoored teachers via OpenBackdoor
```

## Recompute the result tables

Download and extract the Zenodo archive (version 2.0.0), then from its root:

```bash
python -B -X utf8 scripts/tifs/analyze.py --out recomputed/
```

This rebuilds the tables of `results/tifs/report/` (including the 31 fixed cells) from the
per-input predictions on a CPU. The archive README describes every file.

## Train

```bash
python -m counterfactual_kd run configs/tifs/<run_id>.yaml --run-id <run_id>
```

Training needs GPUs; datasets and base models are downloaded from the Hugging Face Hub. Model
weights are not distributed; backdoored checkpoints are available to researchers on request.

## Versions

- **v2.0.0** (November 2026): the version-2 experiments and evaluation protocol. It corrects
  defects of v1 (among them, the configured student learning rate was not passed to the
  distillation trainer, and SST-2 evaluation inputs overlapped the teachers'
  checkpoint-selection data) and supersedes v1 conclusions, including the v1 "no transfer"
  result for ATBA. `RELEASE_NOTES_v2.md` in the Zenodo archive lists all changes.
- **v1.0 / v1.1** (commits `d58c87e`, `3aced28`; Zenodo
  [10.5281/zenodo.21782934](https://doi.org/10.5281/zenodo.21782934) and
  [10.5281/zenodo.23105433](https://doi.org/10.5281/zenodo.23105433)): Cohen's h with
  paired-bootstrap intervals and ROPE verdicts.

## License

Code under the MIT License (`LICENSE`). The data in the Zenodo archive are under CC BY 4.0.
