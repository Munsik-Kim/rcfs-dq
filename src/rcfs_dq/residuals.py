"""Same-state one-stage endpoints with explicit subtraction precision."""

from dataclasses import dataclass

import torch

from .quantization import named_modules_fake_quantized
from .verification import tensor_checksum


def endpoint_difference(candidate, baseline, *, arithmetic="fp64"):
    if candidate.shape != baseline.shape:
        raise ValueError("Endpoint shapes differ")
    if not torch.isfinite(candidate).all() or not torch.isfinite(baseline).all():
        raise ValueError("Nonfinite endpoint")
    if arithmetic == "fp64":
        return candidate.double() - baseline.double()
    if arithmetic == "source_native":
        # Executable convention of the published native matched-cost study.
        return (candidate.float() - baseline.float()).double()
    raise ValueError("Unknown difference arithmetic")


def squared_energy(residual):
    if residual.ndim < 2 or not torch.isfinite(residual).all():
        raise ValueError("Finite residual with batch dimension required")
    return torch.linalg.vector_norm(residual.double().flatten(1), dim=1).square()


@dataclass(frozen=True)
class ResidualObservation:
    baseline: torch.Tensor
    quantized: torch.Tensor
    residual: torch.Tensor
    restored: bool


@torch.inference_mode()
def extract_stage_residual(model, update, state, module_names, bits, *, arithmetic="fp64"):
    """update(state) must be a deterministic stateless one-step callable.

    The caller owns stage/conditioning/scheduler cloning. Neither output is
    fed into the other arm. Weights are restored even when update raises.
    """
    before = tensor_checksum(state)
    baseline = update(state.detach().clone()).detach().clone()
    with named_modules_fake_quantized(model, module_names, bits) as stats:
        quantized = update(state.detach().clone()).detach().clone()
    if tensor_checksum(state) != before:
        raise RuntimeError("Observer mutated the incoming state")
    return ResidualObservation(
        baseline,
        quantized,
        endpoint_difference(quantized, baseline, arithmetic=arithmetic),
        stats.before_checksums == stats.after_checksums,
    )
