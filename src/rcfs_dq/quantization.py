"""Per-output-channel weight-only fake quantization; no packed kernels."""

import contextlib
from dataclasses import dataclass
from typing import Iterator

import torch

from .verification import tensor_checksum


@dataclass(frozen=True)
class QuantizedWeight:
    value: torch.Tensor
    saturation_count: int
    element_count: int
    zero_scale_channels: int


def fake_quantize_per_output_channel(weight: torch.Tensor, bits: int | None) -> QuantizedWeight:
    if weight.ndim < 2 or not weight.is_floating_point():
        raise ValueError("A floating-point weight with output-channel dimension is required")
    if bits is not None and (type(bits) is not int or not 2 <= bits <= 16):
        raise ValueError("bits must be an integer in [2,16] or None")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains NaN or Inf")
    if bits is None or bits >= 16:
        return QuantizedWeight(weight.detach().clone(), 0, weight.numel(), 0)
    original_dtype = weight.dtype
    work = weight.detach().float()
    flat = work.reshape(work.shape[0], -1)
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    absmax = flat.abs().amax(dim=1, keepdim=True)
    zero = absmax == 0
    scale = torch.where(zero, torch.ones_like(absmax), absmax / qmax)
    integer_unclipped = torch.round(flat / scale)
    saturation = int(((integer_unclipped < qmin) | (integer_unclipped > qmax)).sum().item())
    integer = integer_unclipped.clamp(qmin, qmax)
    quantized = integer * scale
    quantized = torch.where(zero, torch.zeros_like(quantized), quantized)
    return QuantizedWeight(
        quantized.reshape_as(work).to(original_dtype),
        saturation,
        weight.numel(),
        int(zero.sum().item()),
    )


def named_modules_selected(
    model: torch.nn.Module, module_names: list[str]
) -> list[tuple[str, torch.nn.Module]]:
    if not module_names or len(module_names) != len(set(module_names)):
        raise ValueError("Nonempty unique module names required")
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise ValueError(f"unknown module names: {missing}")
    selected = [(name, modules[name]) for name in module_names]
    if any(not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) for _, module in selected):
        raise ValueError("all selected modules must expose quantizable weights")
    if len({module.weight.data_ptr() for _, module in selected}) != len(selected):
        raise ValueError("Shared weight storage cannot be quantized twice")
    return selected


@dataclass
class QuantizationContextStats:
    module_names: list[str]
    saturation_count: int
    element_count: int
    zero_scale_channels: int
    before_checksums: dict[str, str]
    after_checksums: dict[str, str]

    @property
    def saturation_rate(self) -> float:
        return self.saturation_count / max(self.element_count, 1)


@contextlib.contextmanager
def named_modules_fake_quantized(
    model: torch.nn.Module, module_names: list[str], bits: int | None
) -> Iterator[QuantizationContextStats]:
    modules = named_modules_selected(model, module_names)
    backups = {name: module.weight.detach().clone() for name, module in modules}
    before = {name: tensor_checksum(value) for name, value in backups.items()}
    stats = QuantizationContextStats([name for name, _ in modules], 0, 0, 0, before, {})
    try:
        with torch.no_grad():
            for _, module in modules:
                quantized = fake_quantize_per_output_channel(module.weight, bits)
                module.weight.copy_(quantized.value)
                stats.saturation_count += quantized.saturation_count
                stats.element_count += quantized.element_count
                stats.zero_scale_channels += quantized.zero_scale_channels
        yield stats
    finally:
        with torch.no_grad():
            for name, module in modules:
                module.weight.copy_(backups[name])
        stats.after_checksums = {name: tensor_checksum(module.weight) for name, module in modules}
        if stats.after_checksums != stats.before_checksums:
            raise RuntimeError("full-precision weights were not restored exactly")
