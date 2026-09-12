from __future__ import annotations

import torch


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.detach().double())


def _match_norm(direction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_norm = _norm(target)
    if target_norm == 0:
        return torch.zeros_like(target)
    out = direction.to(device=target.device, dtype=target.dtype).clone()
    if _norm(out) == 0:
        raise ValueError("control direction has zero norm")
    for _ in range(8):
        out = (out.double() * (target_norm / _norm(out))).to(target.dtype)
    # A global fp16/bf16 rescale can stall one ULP away from the requested norm.
    # Correct one entry whose representable squared magnitude best closes the gap.
    flat = out.reshape(-1)
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
    relative = abs(float(_norm(out) - target_norm)) / max(float(target_norm), 1e-30)
    if relative > 1e-6:
        raise RuntimeError(f"control norm mismatch: relative={relative}")
    return out


def isotropic(eta: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=eta.device).manual_seed(int(seed))
    direction = torch.randn(eta.shape, generator=generator, device=eta.device, dtype=torch.float32)
    return _match_norm(direction, eta)


def seed_shuffled(eta: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    return _match_norm(donor, eta)


def sign_randomized(eta: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=eta.device).manual_seed(int(seed))
    signs = (
        torch.randint(0, 2, eta.shape, generator=generator, device=eta.device, dtype=torch.int8)
        .mul_(2)
        .sub_(1)
    )
    return eta.abs() * signs.to(eta.dtype)


def channel_shuffled(eta: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[int]]:
    generator = torch.Generator(device=eta.device).manual_seed(int(seed))
    permutation = torch.randperm(eta.shape[1], generator=generator, device=eta.device)
    return eta[:, permutation], permutation.detach().cpu().tolist()


def relative_norm_error(control: torch.Tensor, target: torch.Tensor) -> float:
    return abs(float(_norm(control) - _norm(target))) / max(float(_norm(target)), 1e-30)
