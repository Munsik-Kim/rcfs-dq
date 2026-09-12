from __future__ import annotations

import math

import torch


def _validate_tensor(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or not x.numel():
        raise ValueError("A nonempty floating-point tensor is required")
    if not torch.isfinite(x).all():
        raise ValueError("Control inputs must be finite")


def _norm(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x.detach().double())
    if not torch.isfinite(norm) or (norm == 0 and torch.count_nonzero(x) != 0):
        raise ValueError("Control norm is outside the supported float64 range")
    return norm


def _match_norm(direction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_tensor(target)
    _validate_tensor(direction)
    if direction.shape != target.shape:
        raise ValueError("Control and target shapes must match exactly")
    target_norm = _norm(target)
    if target_norm == 0:
        return torch.zeros_like(target, memory_format=torch.contiguous_format)
    if not torch.isfinite(target_norm.square()) or target_norm.square() == 0:
        raise ValueError("Target energy is outside the supported float64 range")
    # The correction below must edit the returned tensor, not a reshape copy.
    out = direction.to(device=target.device, dtype=target.dtype).clone(
        memory_format=torch.contiguous_format
    )
    _validate_tensor(out)
    if _norm(out) == 0:
        raise ValueError("control direction has zero norm")
    for _ in range(8):
        out = (out.double() * (target_norm / _norm(out))).to(target.dtype)
        _validate_tensor(out)
        if _norm(out) == 0:
            raise ValueError("Rescaled control underflows the target dtype")
    # A global fp16/bf16 rescale can stall one ULP away from the requested norm.
    # Correct one entry whose representable squared magnitude best closes the gap.
    flat = out.view(-1)
    for _ in range(4):
        current_sq = _norm(out).square()
        difference = target_norm.square() - current_sq
        values = flat.double()
        desired_sq = values.square() + difference
        valid = desired_sq >= 0
        candidates = torch.sqrt(desired_sq.clamp_min(0)).to(out.dtype).double()
        candidates = torch.copysign(candidates, values)
        resulting_sq = current_sq - values.square() + candidates.square()
        errors = (resulting_sq - target_norm.square()).abs()
        errors = torch.where(valid, errors, torch.full_like(errors, torch.inf))
        best = int(errors.argmin())
        flat[best] = candidates[best].to(out.dtype)
    _validate_tensor(out)
    relative = abs(float(_norm(out) - target_norm)) / float(target_norm)
    if not math.isfinite(relative):
        raise ValueError("Nonfinite control norm error")
    if relative > 1e-6:
        raise RuntimeError(f"control norm mismatch: relative={relative}")
    return out


def isotropic(eta: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a contiguous norm-matched direction; never modify the input."""
    _validate_tensor(eta)
    generator = torch.Generator(device=eta.device).manual_seed(int(seed))
    direction = torch.randn(eta.shape, generator=generator, device=eta.device, dtype=torch.float32)
    return _match_norm(direction, eta)


def seed_shuffled(eta: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    """Match a same-shaped finite donor, returning independent contiguous storage."""
    return _match_norm(donor, eta)


def sign_randomized(eta: torch.Tensor, seed: int) -> torch.Tensor:
    _validate_tensor(eta)
    generator = torch.Generator(device=eta.device).manual_seed(int(seed))
    signs = (
        torch.randint(0, 2, eta.shape, generator=generator, device=eta.device, dtype=torch.int8)
        .mul_(2)
        .sub_(1)
    )
    return eta.abs() * signs.to(eta.dtype)


def channel_shuffled(eta: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[int]]:
    _validate_tensor(eta)
    if eta.ndim < 2:
        raise ValueError("Channel shuffling requires dimension 1")
    generator = torch.Generator(device=eta.device).manual_seed(int(seed))
    permutation = torch.randperm(eta.shape[1], generator=generator, device=eta.device)
    return eta[:, permutation], permutation.detach().cpu().tolist()


def relative_norm_error(control: torch.Tensor, target: torch.Tensor) -> float:
    _validate_tensor(control)
    _validate_tensor(target)
    if control.shape != target.shape:
        raise ValueError("Control and target shapes must match exactly")
    error = abs(float(_norm(control) - _norm(target))) / max(float(_norm(target)), 1e-30)
    if not math.isfinite(error):
        raise ValueError("Nonfinite control norm error")
    return error
