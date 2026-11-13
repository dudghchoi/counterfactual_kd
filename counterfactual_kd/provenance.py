"""Provenance fingerprints (TIFS addendum A15; audit findings S2 and F5).

Stages skip work when an output file exists, so a result can silently
come from an older checkpoint or older code. These helpers record what an
output was made from: the SHA-256 of a checkpoint's weight files, a
canonical SHA-256 of its tensors (independent of the file container and
sharding), the teacher epoch, and a fingerprint of the package source.

Used only behind default-off flags (``stage1.record_provenance``,
``stage2.record_provenance``, ``stage0.save_every_epoch``). Stdlib apart
from the optional ``safetensors``/``torch`` imports of
``tensors_sha256``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

WEIGHT_PATTERNS = ("*.safetensors", "*.bin")
_PKG = os.path.dirname(os.path.abspath(__file__))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_files(ckpt_dir: str) -> list[str]:
    """Weight files of an HF checkpoint dir (not recursive), sorted."""
    out: set[str] = set()
    for pat in WEIGHT_PATTERNS:
        out.update(glob.glob(os.path.join(ckpt_dir, pat)))
    return sorted(out)


def weights_sha256(ckpt_dir: str) -> dict[str, str]:
    """{file name: SHA-256} of the checkpoint's weight files."""
    return {os.path.basename(p): file_sha256(p) for p in weight_files(ckpt_dir)}


def combined_sha256(file_hashes: dict[str, str]) -> str:
    """One hash over ``{name: sha}`` (sorted ``<sha>  <name>`` lines)."""
    lines = "".join(f"{v}  {k}\n" for k, v in sorted(file_hashes.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def tensors_sha256(ckpt_dir: str) -> str:
    """Canonical SHA-256 of a safetensors checkpoint's tensors: names in
    sorted order, each with dtype, shape and raw bytes. Equal for equal
    weights whatever the sharding or file metadata."""
    from safetensors import safe_open
    import torch
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {ckpt_dir}")
    import contextlib
    h = hashlib.sha256()
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(safe_open(f, framework="pt")) for f in files]
        where = {k: sf for sf in handles for k in sf.keys()}
        for k in sorted(where):
            t = where[k].get_tensor(k).contiguous()
            h.update(k.encode() + b"\0" + str(t.dtype).encode() + b"\0"
                     + str(tuple(t.shape)).encode() + b"\0")
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


_FINGERPRINTS: dict[str, dict] = {}


def code_fingerprint(root: str = _PKG) -> dict:
    """SHA-256 over every ``.py`` file of the package (sorted
    ``<sha>  <relative path>`` lines, as scripts/tifs/snapshot_code.py
    lists them). Computed once per process (first call; the A15 entry
    points call it when they start), so a later code edit is not
    attributed to a process that loaded the earlier code."""
    if root in _FINGERPRINTS:
        return dict(_FINGERPRINTS[root])
    hashes = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for name in files:
            if name.endswith(".py"):
                full = os.path.join(d, name)
                rel = os.path.relpath(full, os.path.dirname(root))
                hashes[rel] = file_sha256(full)
    _FINGERPRINTS[root] = {"code_sha256": combined_sha256(hashes),
                           "code_scope": f"{os.path.basename(root)}/**/*.py",
                           "code_n_files": len(hashes)}
    return dict(_FINGERPRINTS[root])


def texts_sha256(texts: list[str]) -> str:
    """SHA-256 of a list of input texts (JSON, UTF-8, order kept)."""
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest()


_TEACHER_CACHE: dict[str, dict] = {}


def teacher_record(teacher_path: str | None) -> dict | None:
    """What a student was distilled from: path, weight-file hashes and the
    epoch, read from the sidecars Stage 0 writes with
    ``stage0.save_every_epoch`` (``epoch_meta.json`` in an epoch dir,
    ``epochs.json`` next to an OB-selected checkpoint). Cached per path."""
    if teacher_path is None:
        return None
    real = os.path.realpath(teacher_path)
    if real in _TEACHER_CACHE:
        return _TEACHER_CACHE[real]
    files = weights_sha256(real)
    rec = {"teacher_path": teacher_path, "teacher_realpath": real,
           "teacher_weights_sha256": files,
           "teacher_weights_sha256_combined": combined_sha256(files),
           "teacher_epoch": None, "teacher_epoch_source": "not recorded"}
    em = os.path.join(real, "epoch_meta.json")
    ej = os.path.join(real, "epochs.json")
    if os.path.exists(em):
        with open(em) as f:
            m = json.load(f)
        rec.update(teacher_epoch=m.get("epoch"), teacher_epoch_source="epoch_meta.json",
                   teacher_tensors_sha256=m.get("tensors_sha256"))
    elif os.path.exists(ej):
        with open(ej) as f:
            m = json.load(f)
        rec.update(teacher_epoch=m.get("ob_selected_epoch"),
                   teacher_epoch_source="epochs.json (OpenBackdoor-selected)",
                   teacher_tensors_sha256=m.get("root_tensors_sha256"))
    _TEACHER_CACHE[real] = rec
    return rec
