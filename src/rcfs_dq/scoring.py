"""Frozen scores only. No fitting, calibration, or outcome input."""

import math
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FrozenCandidateB:
    geometry: dict
    beta: float

    def __post_init__(self):
        values = dict(self.geometry)
        if not values or any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Frozen geometry must be positive and finite")
        if not math.isfinite(self.beta):
            raise ValueError("Finite frozen beta required")
        object.__setattr__(self, "geometry", MappingProxyType(values))

    def score(self, cell_id, energy):
        if not math.isfinite(energy) or energy < 0:
            raise ValueError("Finite nonnegative energy required")
        value = energy * self.geometry[cell_id] ** self.beta
        if not math.isfinite(value):
            raise ValueError("Nonfinite score")
        return value
