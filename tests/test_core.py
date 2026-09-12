import copy

import numpy as np
import pytest
import torch

from rcfs_dq.controls import channel_shuffled, isotropic, relative_norm_error, seed_shuffled
from rcfs_dq.decisions import Candidate, class_block_ci, evaluate_regret, exact_cost_sets, select
from rcfs_dq.geometry import central_derivative, finite_horizon_response, lcg_ratio_of_sums
from rcfs_dq.quantization import fake_quantize_per_output_channel, named_modules_fake_quantized
from rcfs_dq.residuals import endpoint_difference, extract_stage_residual, squared_energy
from rcfs_dq.scoring import FrozenCandidateB
from rcfs_dq.verification import assert_close, independent_regret, verify_inventory


@pytest.mark.parametrize("bits", [4, 6, 8, 16, None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantizer_against_independent_numpy(bits, dtype):
    w = torch.tensor([[0, 0, 0], [-0.7, 0.3, 0.9]], dtype=dtype)
    before = w.clone()
    q = fake_quantize_per_output_channel(w, bits)
    assert q.value.dtype == dtype and torch.equal(w, before)
    if bits is None or bits == 16:
        assert torch.equal(q.value, w)
    else:
        a = w.float().numpy()
        scale = np.max(abs(a), axis=1, keepdims=True) / np.float32(2 ** (bits - 1) - 1)
        scale[scale == 0] = 1
        reference = np.clip(np.rint(a / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * scale
        assert torch.equal(q.value, torch.tensor(reference).to(dtype))
        assert q.zero_scale_channels == 1


@pytest.mark.parametrize("bad", [1, 17, True, 4.0])
def test_bad_precision(bad):
    with pytest.raises(ValueError):
        fake_quantize_per_output_channel(torch.ones(2, 3), bad)


def test_nonfinite_identity_rejected():
    with pytest.raises(ValueError):
        fake_quantize_per_output_channel(torch.full((2, 3), float("nan")), None)


def test_restore_even_on_exception():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="intentional"):
        with named_modules_fake_quantized(model, ["0"], 4):
            raise RuntimeError("intentional")
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("names", [[], ["0", "0"], ["missing"]])
def test_bad_module_scope(names):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    with pytest.raises(ValueError):
        with named_modules_fake_quantized(model, names, 4):
            pass


def test_observer_and_identity_residual():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).eval()
    x = torch.tensor([[0.2, 0.4, 0.7, 1.1]])
    before = x.clone()
    obs = extract_stage_residual(model, model, x, ["0"], None)
    assert obs.restored and torch.equal(x, before)
    assert torch.count_nonzero(obs.residual) == 0
    assert torch.equal(obs.baseline, obs.quantized)


def test_source_subtract_convention_and_shapes():
    a = torch.tensor([[1.0, 2048.0]], dtype=torch.float16)
    b = torch.tensor([[0.0001, -2048.0]], dtype=torch.float16)
    ref = (a.float() - b.float()).double()
    assert torch.equal(endpoint_difference(a, b, arithmetic="source_native"), ref)
    assert squared_energy(ref).shape == (1,)
    assert not torch.equal((a - b).double(), endpoint_difference(a, b))
    with pytest.raises(ValueError):
        endpoint_difference(a, b[:, :1])


def test_controls_preserve_norm_and_donor():
    x = torch.arange(1, 33, dtype=torch.float32).reshape(1, 2, 4, 4)
    donor = x.flip(-1).clone()
    before = donor.clone()
    shuffled = seed_shuffled(x, donor)
    assert torch.equal(donor, before)
    assert relative_norm_error(shuffled, x) < 1e-6
    assert relative_norm_error(isotropic(x, 9), x) < 1e-6
    c, permutation = channel_shuffled(x, 9)
    assert relative_norm_error(c, x) < 1e-6 and sorted(permutation) == [0, 1]
    assert torch.equal(isotropic(x, 9), isotropic(x, 9))


def test_zero_control():
    x = torch.zeros(1, 1, 2, 2)
    assert torch.equal(isotropic(x, 0), x)


def test_linear_gain_ratio_of_sums():
    u = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    x = torch.tensor([[3.0, 1.0]], dtype=torch.float64)

    def f(z):
        return z * torch.tensor([[2.0, 3.0]])

    r = finite_horizon_response(f, x, u, 0.125)
    assert torch.equal(r["derivative"], torch.tensor([[2.0, 6.0]], dtype=torch.float64))
    assert lcg_ratio_of_sums([2.0, 6.0], [1.0, 2.0]) == 8.0
    assert lcg_ratio_of_sums([2.0, 6.0], [1.0, 2.0]) != np.mean([4.0, 9.0])


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_invalid_central_scale(alpha):
    with pytest.raises(ValueError):
        central_derivative(torch.ones(1, 2), torch.zeros(1, 2), alpha)


def test_zero_gain_denominator_is_undefined():
    with pytest.raises(ValueError):
        lcg_ratio_of_sums([1], [0])


def candidates():
    return [Candidate(f"c{k}", "g", k, 4, 10) for k in [4, 10, 15]]


def test_exact_cost_no_tolerance():
    c = candidates()
    assert len(exact_cost_sets(c)) == 1
    wrong = Candidate("other", "g", 19, 4, 11)
    assert len(next(iter(exact_cost_sets(c + [wrong]).values()))) == 3
    with pytest.raises(ValueError):
        evaluate_regret([c[0], wrong], {c[0].cell_id: 0, "other": 1}, "other")


def test_tie_oracle_and_independent_regret():
    c = candidates()
    y = {x.cell_id: (0 if x.stage < 15 else 2) for x in c}
    assert select(c, y) == "c4"
    r = evaluate_regret(c, y, "c15")
    v = independent_regret(y, "c15", ["c4", "c10", "c15"])
    assert r["oracle"] == v["oracle"] == "c4" and r["normalized_regret"] == 1


def test_zero_range_preserves_absolute_regret():
    c = candidates()
    r = evaluate_regret(c, {x.cell_id: 3.0 for x in c}, "c15")
    assert not r["resolved"] and r["normalized_regret"] is None and r["absolute_regret"] == 0


def test_hybrid_guard_orientation():
    c = [Candidate("high", "g1", 4, 4, 10, regime="M3"), Candidate("low", "g2", 10, 4, 10)]
    assert select(c, {"high": 0.01, "low": 1}, hybrid=True) == "low"


def test_frozen_score_no_mutation():
    data = {"a": 4.0}
    model = FrozenCandidateB(data, 0.5)
    data["a"] = 9
    assert model.score("a", 2.0) == 4
    with pytest.raises(TypeError):
        model.geometry["a"] = 10
    with pytest.raises(ValueError):
        model.score("a", float("nan"))


def test_paired_bootstrap_class_units():
    idx = np.random.Generator(np.random.PCG64(1)).integers(0, 4, (100, 4))
    difference = np.array([1.0, 2.0, 3.0, 4.0]) - np.array([2.0, 3.0, 4.0, 5.0])
    out = class_block_ci(difference, indices=idx)
    assert set(out.values()) == {-1.0}


def test_manifest_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        verify_inventory(tmp_path, {"files": [{"path": "../escape"}]})


def test_float_tolerance_and_nonfinite():
    assert_close(1.0, 1.0 + 1e-13)
    with pytest.raises(AssertionError):
        assert_close(float("nan"), float("nan"))


def test_inplace_continuation_preserves_injection():
    x = torch.ones(1, 3, dtype=torch.float64)
    before = x.clone()

    def twice_inplace(value):
        return value.mul_(2)

    result = finite_horizon_response(twice_inplace, x, x, 0.125)
    assert torch.equal(result["realized_plus"], torch.full_like(x, 0.125))
    assert torch.equal(result["realized_minus"], torch.full_like(x, -0.125))
    assert torch.equal(result["derivative"], 2 * x)
    assert torch.equal(x, before)


def test_duplicate_decision_candidate_rejected():
    c = candidates()[0]
    with pytest.raises(ValueError):
        select([c, c], {c.cell_id: 1.0})


def test_empty_bootstrap_rejected():
    with pytest.raises(ValueError):
        class_block_ci([1.0], indices=np.empty((0, 1), dtype=int))


def test_inventory_content_mismatch(tmp_path):
    path = tmp_path / "record.txt"
    path.write_text("data")
    with pytest.raises(AssertionError):
        verify_inventory(
            tmp_path, {"files": [{"path": "record.txt", "bytes": 4, "sha256": "0" * 64}]}
        )
