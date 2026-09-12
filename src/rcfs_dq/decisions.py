"""Exact theoretical packed-saving sets and native-dtype selection regret."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    cell_id: str
    group: str
    stage: int
    bits: int
    numel: int
    group_order: int = 0
    regime: str = "L78"

    def __post_init__(self):
        if type(self.numel) is not int or self.numel <= 0:
            raise ValueError("Exact positive integer numel required")
        if type(self.bits) is not int or not 2 <= self.bits <= 16:
            raise ValueError("Integer weight precision required")
        if type(self.stage) is not int or self.stage < 0 or self.regime not in {"M3", "L78"}:
            raise ValueError("Invalid stage or frozen regime")

    @property
    def packed_bits(self):
        return self.numel * self.bits

    @property
    def saving_bits(self):
        return self.numel * (16 - self.bits)

    @property
    def tie_key(self):
        return self.stage, self.group_order, -self.bits, self.cell_id


def exact_cost_sets(candidates, *, primary=True):
    groups = defaultdict(list)
    ids = [c.cell_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate")
    for c in candidates:
        key = (c.group, c.bits, c.saving_bits) if primary else (c.saving_bits,)
        groups[key].append(c)
    return {k: tuple(sorted(v, key=lambda c: c.tie_key)) for k, v in groups.items() if len(v) >= 2}


def select(candidates, scores, *, hybrid=False):
    if len({c.cell_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate")
    if not candidates or set(scores) != {c.cell_id for c in candidates}:
        raise ValueError("Exact score/candidate coverage required")
    if any(not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("Finite nonnegative scores required")
    return min(
        candidates,
        key=lambda c: (int(c.regime == "M3") if hybrid else 0, scores[c.cell_id], *c.tie_key),
    ).cell_id


def evaluate_regret(candidates, values, selected):
    if len(candidates) < 2 or len({c.saving_bits for c in candidates}) != 1:
        raise ValueError("At least two exact matched-saving candidates required")
    oracle = select(candidates, values)
    if selected not in values:
        raise ValueError("Selected candidate missing")
    span = max(values.values()) - min(values.values())
    tau = 1e-12 * max(max(values.values()), 1.0)
    regret = values[selected] - values[oracle]
    return dict(
        oracle=oracle,
        absolute_regret=regret,
        normalized_regret=regret / span if span > tau else None,
        resolved=span > tau,
        risk_range=span,
        range_tau=tau,
    )


def class_block_ci(class_values, *, indices):
    """Caller first forms equal-set/seed class means; shared indices preserve pairing."""
    values = np.asarray(class_values, dtype=np.float64)
    indices = np.asarray(indices)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Finite class means required")
    if (
        indices.ndim != 2
        or not indices.size
        or indices.shape[1] != len(values)
        or indices.dtype.kind not in "iu"
    ):
        raise ValueError("Integer class-block index matrix required")
    if indices.min() < 0 or indices.max() >= len(values):
        raise ValueError("Out-of-range bootstrap index")
    draws = values[indices].mean(axis=1)
    lo, hi, lower, upper = np.quantile(draws, [0.025, 0.975, 0.05, 0.95], method="linear")
    return dict(
        point=float(values.mean()),
        low=float(lo),
        high=float(hi),
        lower95=float(lower),
        upper95=float(upper),
    )
