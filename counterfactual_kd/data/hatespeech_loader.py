"""
Vicomtech HateSpeech loader (de Gibert et al., ALW2 2018; W18-5102).

Source repo cloned to ``data/external/vicomtech/``:
    - sampled_train/<file_id>.txt   — balanced training samples
    - sampled_test/<file_id>.txt    — balanced test samples
    - all_files/<file_id>.txt       — full corpus (unbalanced)
    - annotations_metadata.csv      — file_id → {label, num_contexts, ...}

This loader returns the Counterfactual-KD standard text-classification format:
    [{"text": str, "label": int}]   with 0=noHate, 1=hate

Design choices (committed 2026-04-29 alongside the Counterfactual-KD Round-2 dataset
matrix):

* **Source: ``all_files/`` (full corpus).** The upstream repo's
  ``sampled_{train,test}`` are a balanced 2.4k subset. The 2026-04-29
  matrix design uses fixed split sizes train=7703 / dev=1000 /
  test=2000 (sum 10703 = the corpus minus the ``idk/skip`` and
  ``relation`` labels). Records with labels other than ``hate`` /
  ``noHate`` are dropped.

* **Class imbalance is intentional (natural distribution).**
  noHate dominates (~88.8%) in ``all_files``. With target_label=0
  (noHate) the BSR evaluation only inserts triggers into hate-labeled
  sentences (``BaseAttack.poison_eval`` filters by non-target), so the
  imbalance affects CACC interpretation — not ASR/BSR. The
  majority-class CACC baseline is ~88.8%; sanity check trained model
  CACC against this when reading results.

* **num_contexts filter is optional (default: keep all).** Sentences
  with num_contexts > 0 cannot be judged without prior context, but
  with the design-locked 1% poison rate dropping them shrinks the
  corpus too far. ``load_hatespeech(num_contexts_filter=0)`` keeps the
  ablation-only single-sentence subset; in that mode the fixed split
  sizes are clamped to whatever the filter leaves.

* **Train/dev/test split derived deterministically.** all_files is
  shuffled with seed=42 then sliced by the fixed sizes; rerunning the
  loader with the same seed yields the same partition.

* **target_label = 0 (noHate).** See ``DatasetSpec("hatespeech")`` in
  counterfactual_kd/registry.py — the safety-bypass framing follows the threat
  model the original authors describe.

* **50/50 balanced subsample (``mode="5050"``).** Used by the
  ``hatespeech_5050`` registry entry as a §4.3.4(a) b-clean precision
  control: drop class-prior $b$ from the natural 0.89 to 0.50 with
  attack/trigger/teacher/student otherwise identical. Construction is
  deterministic at seed=42:

      1. Collect all ``hate``-labeled records (n=1196 in all_files).
      2. Sample ``noHate`` records uniformly without replacement until
         |noHate|=|hate|, giving |hate|+|noHate|=2392 records total.
      3. Shuffle the balanced corpus with the same seed.
      4. Partition with the natural-split shape (7703/1000/2000)
         scaled proportionally to total=2392 — see ``DEFAULT_SIZES_5050``.

  This guarantees ≈50/50 class balance in *each* split (train/dev/test)
  because the shuffle-then-slice operates on a corpus that's already
  exactly 50/50; per-split deviations are sampling noise on small splits
  only.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Optional


DEFAULT_ROOT = "data/external/vicomtech"
_LABEL_MAP: dict[str, int] = {"noHate": 0, "hate": 1}


def _read_metadata(root: str) -> dict[str, dict]:
    """Parse ``annotations_metadata.csv`` into a {file_id: row} dict."""
    path = os.path.join(root, "annotations_metadata.csv")
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["file_id"]] = {
                "label": row["label"],
                "num_contexts": int(row["num_contexts"]),
                "user_id": row["user_id"],
                "subforum_id": row["subforum_id"],
            }
    return out


def _load_split_dir(root: str, split_dir: str,
                    metadata: dict[str, dict],
                    num_contexts_filter: Optional[int]) -> list[dict]:
    """Walk a sampled_{train,test} directory and emit standard records.

    Skips files whose metadata is missing, whose label is not in
    {noHate, hate} (the upstream repo also has small "idk" / "relation"
    tags), or — if ``num_contexts_filter`` is given — whose
    num_contexts exceeds it.
    """
    out: list[dict] = []
    base = os.path.join(root, split_dir)
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".txt"):
            continue
        file_id = fname[:-4]  # strip ".txt"
        meta = metadata.get(file_id)
        if meta is None:
            continue
        if num_contexts_filter is not None and \
                meta["num_contexts"] > num_contexts_filter:
            continue
        if meta["label"] not in _LABEL_MAP:
            continue
        with open(os.path.join(base, fname), encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            continue
        out.append({"text": text, "label": _LABEL_MAP[meta["label"]]})
    return out


# Matrix-locked split sizes (design table 2026-04-29).
DEFAULT_SIZES: tuple[int, int, int] = (7703, 1000, 2000)

# 50/50 mode: scaled proportionally from DEFAULT_SIZES to the maximum
# balanced corpus size (2 × n_hate = 2 × 1196 = 2392). Held as a constant
# so callers / tests can introspect the expected counts. Rounded with
# trailing absorption: dev/test get floor, train absorbs the remainder
# so the three sum to exactly 2 × n_hate.
#
# Computation: total_5050 = 2 * 1196 = 2392
#   train ≈ 7703/10703 * 2392 = 1721.41 → 1721
#   dev   ≈ 1000/10703 * 2392 =  223.49 → 223
#   test  ≈ 2000/10703 * 2392 =  446.98 → 447
#   train absorbs +1 rounding remainder → (1722, 223, 447), sum=2392.
DEFAULT_SIZES_5050: tuple[int, int, int] = (1722, 223, 447)


def _balance_5050(corpus: list[dict], seed: int) -> list[dict]:
    """Return a balanced 50/50 subset of ``corpus``.

    Takes every ``hate`` record (label==1) and an equal-count
    deterministic sample of ``noHate`` (label==0), then shuffles the
    combined set with the same ``seed`` so train/dev/test slices each
    inherit ≈50/50 balance.

    Caller must have already filtered the corpus to {noHate, hate} only
    (which ``_load_corpus`` does).
    """
    hate = [r for r in corpus if r["label"] == 1]
    nohate = [r for r in corpus if r["label"] == 0]
    rng = random.Random(seed)
    # Sample noHate without replacement; if there are fewer noHate than
    # hate (won't happen for Vicomtech, but defend anyway) use whatever
    # the smaller class allows.
    n_pick = min(len(hate), len(nohate))
    nohate_picked = rng.sample(nohate, n_pick)
    hate_picked = hate[:n_pick]  # keep ordering deterministic
    balanced = hate_picked + nohate_picked
    rng.shuffle(balanced)
    return balanced


def _load_corpus(root: str,
                 metadata: dict[str, dict],
                 num_contexts_filter: Optional[int]) -> list[dict]:
    """Load all records from ``all_files/`` matching the filter.

    Drops files whose label is not in {noHate, hate} (the upstream
    corpus also has small ``idk/skip`` and ``relation`` tags that
    don't fit a binary classifier) and — if ``num_contexts_filter``
    is given — files whose num_contexts exceeds it.
    """
    base = os.path.join(root, "all_files")
    out: list[dict] = []
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".txt"):
            continue
        file_id = fname[:-4]
        meta = metadata.get(file_id)
        if meta is None:
            continue
        if num_contexts_filter is not None and \
                meta["num_contexts"] > num_contexts_filter:
            continue
        if meta["label"] not in _LABEL_MAP:
            continue
        with open(os.path.join(base, fname), encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            continue
        out.append({"text": text, "label": _LABEL_MAP[meta["label"]]})
    return out


def load_hatespeech(split: str = "train",
                    *,
                    root: str = DEFAULT_ROOT,
                    seed: int = 42,
                    num_contexts_filter: Optional[int] = None,
                    sizes: Optional[tuple[int, int, int]] = None,
                    mode: str = "natural",
                    ) -> list[dict]:
    """Load the Vicomtech hate-speech dataset.

    Args:
        split:    "train" | "dev" | "test".
        root:     dataset root (default: data/external/vicomtech).
        seed:    shuffle seed used to derive the partition.
        num_contexts_filter:
            None (default) — keep all sentences (the matrix-locked
                             setting; sizes will hit DEFAULT_SIZES).
            0              — single-sentence-judgable items only
                             (ablation; sizes will shrink and the
                             requested ``sizes`` clamp to whatever's
                             left).
            Positive int   — items with num_contexts <= the value.
        sizes: (n_train, n_dev, n_test). When ``None`` (default), uses
            ``DEFAULT_SIZES`` for ``mode='natural'`` and
            ``DEFAULT_SIZES_5050`` for ``mode='5050'``. If the filtered
            corpus is smaller than the sum, sizes are clamped
            proportionally and a warning is printed.
        mode: "natural" (default) — full corpus, natural class prior
                                    (~89/11 noHate/hate).
              "5050"              — balanced 50/50 subsample. Used by
                                    the ``hatespeech_5050`` registry
                                    entry for §4.3.4(a) b-clean
                                    precision control.

    Returns:
        List of {"text": str, "label": int} records.
        Labels: 0 = noHate, 1 = hate. Target backdoor label is 0
        ("Not Harmful" bypass), set in DatasetSpec.

    Raises:
        ValueError on unknown split or unknown mode.
        FileNotFoundError if the dataset hasn't been cloned to ``root``.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Vicomtech hate-speech dataset not found at {root}. "
            f"Clone with: git clone "
            f"https://github.com/Vicomtech/hate-speech-dataset.git {root}"
        )

    if mode not in ("natural", "5050"):
        raise ValueError(
            f"Unknown mode: {mode!r}. Expected 'natural' or '5050'."
        )

    if sizes is None:
        sizes = DEFAULT_SIZES_5050 if mode == "5050" else DEFAULT_SIZES

    metadata = _read_metadata(root)
    corpus = _load_corpus(root, metadata, num_contexts_filter)

    if mode == "5050":
        # Replace ``shuffle(corpus)`` with the balanced construction; the
        # downstream slice logic then partitions the balanced corpus.
        corpus = _balance_5050(corpus, seed)
    else:
        rng = random.Random(seed)
        rng.shuffle(corpus)

    n_train, n_dev, n_test = sizes
    total_requested = n_train + n_dev + n_test
    if total_requested > len(corpus):
        # Filter shrank the corpus below the requested split. Scale
        # proportionally so callers always get *something*.
        scale = len(corpus) / total_requested
        n_train = int(n_train * scale)
        n_dev = int(n_dev * scale)
        n_test = len(corpus) - n_train - n_dev   # absorb rounding
        print(
            f"[hatespeech_loader] Filtered corpus has {len(corpus)} < "
            f"{total_requested} records (mode={mode!r}); scaled split to "
            f"({n_train}, {n_dev}, {n_test})."
        )

    if split == "train":
        return corpus[:n_train]
    if split == "dev":
        return corpus[n_train:n_train + n_dev]
    if split == "test":
        return corpus[n_train + n_dev:n_train + n_dev + n_test]

    raise ValueError(
        f"Unknown split: {split!r}. Expected 'train', 'dev', or 'test'."
    )


def load_hatespeech_5050(split: str = "train",
                         *,
                         root: str = DEFAULT_ROOT,
                         seed: int = 42,
                         num_contexts_filter: Optional[int] = None,
                         sizes: Optional[tuple[int, int, int]] = None,
                         ) -> list[dict]:
    """Convenience wrapper: ``load_hatespeech(..., mode='5050')``.

    Used by ``DatasetSpec("hatespeech_5050")`` (registry.py) and by the
    ``hatespeech_5050`` entry in :data:`counterfactual_kd.data.loaders.DATASET_LOADERS`.
    Returns the balanced 50/50 subsample (see module docstring §50/50).
    """
    return load_hatespeech(split, root=root, seed=seed,
                           num_contexts_filter=num_contexts_filter,
                           sizes=sizes, mode="5050")


def split_sizes(root: str = DEFAULT_ROOT,
                seed: int = 42,
                num_contexts_filter: Optional[int] = None,
                sizes: Optional[tuple[int, int, int]] = None,
                mode: str = "natural",
                ) -> dict[str, int]:
    """Return {split: count} actually delivered for the given settings.

    With ``mode='natural'`` defaults, returns
    ``{train: 7703, dev: 1000, test: 2000}``. With ``mode='5050'``
    defaults, returns ``{train: 1722, dev: 223, test: 447}`` (sum=2392 =
    2 × n_hate). With ``num_contexts_filter=0`` the corpus shrinks and
    the loader proportionally rescales the splits.
    """
    return {
        "train": len(load_hatespeech("train", root=root, seed=seed,
                                     num_contexts_filter=num_contexts_filter,
                                     sizes=sizes, mode=mode)),
        "dev":   len(load_hatespeech("dev", root=root, seed=seed,
                                     num_contexts_filter=num_contexts_filter,
                                     sizes=sizes, mode=mode)),
        "test":  len(load_hatespeech("test", root=root, seed=seed,
                                     num_contexts_filter=num_contexts_filter,
                                     sizes=sizes, mode=mode)),
    }
