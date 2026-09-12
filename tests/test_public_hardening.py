"""Numeric boundary regressions; synthetic inputs are not scientific evidence."""

import copy

import numpy as np
import pytest
import torch

from rcfs_dq.controls import (
    channel_shuffled,
    isotropic,
    relative_norm_error,
    seed_shuffled,
    sign_randomized,
)
from rcfs_dq.geometry import lcg_ratio_of_sums
from rcfs_dq.quantization import fake_quantize_per_output_channel, named_modules_fake_quantized


# Seven fixtures embedded in the public review maintenance contract.
def test_review_noncontiguous_donor():
    torch.manual_seed(0)
    eta = torch.randn((2, 3, 4, 5), dtype=torch.float16)
    donor = torch.randn((2, 3, 5, 4), dtype=torch.float16).transpose(-1, -2)
    actual = seed_shuffled(eta, donor)
    assert torch.equal(actual, seed_shuffled(eta, donor.contiguous()))
    assert relative_norm_error(actual, eta) <= 1e-6


def test_review_shape_mismatch():
    with pytest.raises(ValueError):
        seed_shuffled(torch.ones(1, 2, 2), torch.ones(4))


def test_review_nonfinite_target():
    with pytest.raises(ValueError):
        isotropic(torch.tensor([[float("inf"), 1.0]]), 0)


def test_review_nonfinite_donor():
    with pytest.raises(ValueError):
        seed_shuffled(torch.ones(1, 2), torch.tensor([[float("nan"), 1.0]]))


def test_review_quantizer_fp32_overflow():
    with pytest.raises(ValueError):
        fake_quantize_per_output_channel(torch.tensor([[1e100, 0.0]], dtype=torch.float64), 4)


def test_review_quantizer_scale_underflow():
    with pytest.raises(ValueError):
        fake_quantize_per_output_channel(torch.tensor([[1e-44, 0.0]], dtype=torch.float32), 8)


def test_review_gain_extreme_range():
    assert lcg_ratio_of_sums([1e200], [1e200]) == 1.0
    assert lcg_ratio_of_sums([1e-200], [1e-200]) == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transpose", "strided"])
def test_control_layout_and_input_immutability(dtype, layout):
    torch.manual_seed(0)
    eta = torch.randn((2, 3, 4, 5), dtype=dtype)
    if layout == "transpose":
        donor = torch.randn((2, 3, 5, 4), dtype=dtype).transpose(-1, -2)
    else:
        donor = torch.randn((2, 3, 4, 10), dtype=dtype)[..., ::2]
    old_eta, old_donor = eta.clone(), donor.clone()
    actual = seed_shuffled(eta, donor)
    assert actual.is_contiguous()
    assert actual.shape == eta.shape and actual.dtype == dtype
    assert torch.equal(actual, seed_shuffled(eta, donor.contiguous()))
    assert relative_norm_error(actual, eta) <= 1e-6
    assert torch.equal(eta, old_eta) and torch.equal(donor, old_donor)


@pytest.mark.parametrize(
    "invalid",
    [
        torch.empty(1, 0),
        torch.ones(1, 2, dtype=torch.int64),
        torch.ones(1, 2, dtype=torch.complex64),
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
    ],
)
@pytest.mark.parametrize("control", [isotropic, sign_randomized, channel_shuffled])
def test_control_invalid_inputs(control, invalid):
    with pytest.raises(ValueError):
        control(invalid, 0)


@pytest.mark.parametrize(
    "donor",
    [
        torch.ones(4),
        torch.empty(1, 0),
        torch.ones(1, 2, dtype=torch.int64),
        torch.full((1, 2), float("nan")),
        torch.full((1, 2), float("inf")),
    ],
)
def test_zero_target_still_validates_donor(donor):
    with pytest.raises(ValueError):
        seed_shuffled(torch.zeros(1, 2), donor)


def test_zero_controls_and_channel_dimension():
    zero = torch.zeros(1, 2)
    assert torch.equal(seed_shuffled(zero, torch.ones_like(zero)), zero)
    assert torch.equal(seed_shuffled(zero, zero), zero)
    with pytest.raises(ValueError):
        seed_shuffled(torch.ones_like(zero), zero)
    with pytest.raises(ValueError):
        channel_shuffled(torch.ones(2), 0)


def test_norm_match_overflow_rejected():
    with pytest.raises(ValueError):
        seed_shuffled(
            torch.ones(1, 2, dtype=torch.float16), torch.full((1, 2), 1e100, dtype=torch.float64)
        )
    with pytest.raises(ValueError):
        relative_norm_error(torch.full((1, 2), float("nan")), torch.ones(1, 2))


@pytest.mark.parametrize("bits", [None, 16])
def test_identity_preserves_finite_fp64_extremes(bits):
    x = torch.tensor([[1e100, 1e-200]], dtype=torch.float64)
    y = fake_quantize_per_output_channel(x, bits).value
    assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_quantizer_empty_rejected(shape):
    with pytest.raises(ValueError):
        fake_quantize_per_output_channel(torch.empty(shape), 4)


def test_quantizer_range_failure_restores_earlier_module():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)).double()
    with torch.no_grad():
        model[1].weight.fill_(1e100)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        with named_modules_fake_quantized(model, ["0", "1"], 4):
            pytest.fail("unsupported weight must fail before entering context")
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_gain_extreme_ratio_of_sums(scale):
    assert lcg_ratio_of_sums([2 * scale, 6 * scale], [scale, 2 * scale]) == 8.0
    assert lcg_ratio_of_sums([0.0], [scale]) == 0.0


@pytest.mark.parametrize(
    "a,b",
    [
        ([1], [0]),
        ([1e200], [1e-200]),
        ([1e-200], [1e200]),
        ([np.inf], [1]),
        ([np.nan], [1]),
        ([], []),
    ],
)
def test_gain_undefined_or_unrepresentable_rejected(a, b):
    with pytest.raises(ValueError):
        lcg_ratio_of_sums(a, b)
