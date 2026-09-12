"""Hash, finite-value and independently implemented scalar checks."""

import hashlib
from pathlib import Path, PurePosixPath

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def tensor_checksum(tensor):
    import torch

    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def assert_close(actual, expected, atol=1e-12, rtol=1e-10):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Shape mismatch or nonfinite comparison")
    error = np.abs(a - b)
    tolerance = atol + rtol * np.abs(b)
    if not (error <= tolerance).all():
        raise AssertionError(f"Numerical mismatch: max absolute difference {error.max()}")
    return float(error.max(initial=0))


def verify_inventory(root, manifest):
    root = Path(root).resolve()
    seen = set()
    for row in manifest["files"]:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Unsafe or duplicate inventory path")
        seen.add(str(relative))
        path = root / str(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Inventory path escapes root")
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise AssertionError(f"Inventory mismatch: {relative}")
    return len(seen)


def independent_regret(values, selected, order):
    """Independent reference: no main scoring/decision imports."""
    if set(values) != set(order) or len(order) != len(set(order)):
        raise ValueError("Candidate coverage mismatch")
    numbers = np.asarray([values[k] for k in order], dtype=np.float64)
    if not np.isfinite(numbers).all() or (numbers < 0).any() or selected not in values:
        raise ValueError("Invalid risk or selected candidate")
    minimum = min(numbers)
    oracle = next(k for k in order if values[k] == minimum)
    regret = float(values[selected] - minimum)
    span = float(max(numbers) - minimum)
    resolved = span > 1e-12 * max(float(max(numbers)), 1.0)
    return dict(
        oracle=oracle,
        absolute_regret=regret,
        normalized_regret=regret / span if resolved else None,
        resolved=resolved,
    )
