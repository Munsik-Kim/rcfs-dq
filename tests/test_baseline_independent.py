"""Synthetic tests of an implementation independent from the main baseline audit."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rcfs_dq.baseline_independent import (
    Comparison,
    independent_choices,
    independent_class_means,
    independent_evaluate,
    independent_headroom,
    independent_interval,
    reference_from_sources,
    validate_baseline_contract,
    validate_inventory,
    validate_source_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def members():
    return [
        dict(cell_id=f"c{k}", stage=k, group_order=0, bits=4, regime="L78") for k in (4, 10, 15)
    ]


def test_latest_index_not_timestep_and_feature_only_api():
    registry = members()
    for row, timestep in zip(registry, [999, 499, 0], strict=True):
        row["timestep"] = timestep
    scores = {r["cell_id"]: 1.0 for r in registry}
    choices = independent_choices(registry, scores, scores, 0, "c10")
    assert choices["latest_stage"] == choices["stage_position_only"] == "c15"
    assert choices["candidate_B"] == choices["energy"] == "c4"
    assert choices["pure_G"] == "c4"  # Pure-G is not assumed to alias the latest stage.
    assert choices["constant"] == "c10"
    assert choices["uniform_random_expected"] is None


def test_gain_argmin_alone_does_not_determine_product_choice():
    registry = members()
    energy = dict(c4=1.0, c10=1.0, c15=10.0)
    a = independent_choices(registry, energy, dict(c4=2, c10=2, c15=1), 1, "c4")
    b = independent_choices(registry, energy, dict(c4=20, c10=20, c15=1), 1, "c4")
    assert a["pure_G"] == b["pure_G"] == "c15"
    assert a["candidate_B"] == "c4" and b["candidate_B"] == "c15"


def test_hybrid_guard_then_within_m3_energy():
    registry = members()
    registry[0]["regime"] = "M3"
    choices = independent_choices(
        registry, dict(c4=0.0, c10=2.0, c15=3.0), dict(c4=1, c10=1, c15=1), 1, "c4"
    )
    assert choices["energy"] == "c4" and choices["hybrid"] == "c10"
    for row in registry:
        row["regime"] = "M3"
    choices = independent_choices(
        registry, dict(c4=3, c10=2, c15=1), dict(c4=0.01, c10=1, c15=100), 1, "c4"
    )
    assert choices["hybrid"] == "c15" and choices["candidate_B"] == "c4"


@pytest.mark.parametrize(
    "values",
    [dict(c4=-1, c10=2, c15=3), dict(c4=np.nan, c10=2, c15=3), dict(c4=np.inf, c10=2, c15=3)],
)
def test_invalid_terminal_risk_fails_closed(values):
    with pytest.raises(ValueError):
        independent_evaluate(["c4", "c10", "c15"], values, "c4")


@pytest.mark.parametrize(
    "order,values",
    [(["a", "a"], {"a": 1}), (["a", "b"], {"a": 1}), (["a", "b"], {"a": 1, "b": 2, "c": 3})],
)
def test_missing_duplicate_extra_risk_fails_closed(order, values):
    with pytest.raises(ValueError):
        independent_evaluate(order, values, "a")


def test_uniform_expected_probability_regret_and_two_hit_conventions():
    result = independent_evaluate(["a", "b", "c"], dict(a=1, b=1, c=7), None)
    assert result["selected"] is None and result["oracle"] == "a"
    assert result["absolute_regret"] == pytest.approx(np.mean([0, 0, 6]))
    assert result["normalized_regret"] == pytest.approx(1 / 3)
    assert result["oracle_hit"] == 1 / 3 and result["all_minima_hit"] == 2 / 3
    assert result["selected_probability_json"] == dict(a=1 / 3, b=1 / 3, c=1 / 3)


def test_zero_and_unresolved_span_not_filled():
    result = independent_evaluate(["a", "b"], dict(a=1, b=1), "b")
    assert result["absolute_regret"] == 0 and result["normalized_regret"] is None
    assert not result["resolved"] and result["oracle_hit"] == 0
    result = independent_evaluate(["a", "b"], dict(a=1, b=1 + 1e-13), "b")
    assert result["absolute_regret"] > 0 and result["normalized_regret"] is None


def test_test_best_fixed_is_not_modal_oracle_and_normalized_is_separate():
    # a wins most units, but its rare loss is much larger; normalized spans differ.
    y = np.array([[0, 1], [0, 1], [100, 0]], dtype=float)
    result = independent_headroom(["a", "b"], y)
    assert result["majority_candidate"] == "a"
    assert result["empirical_best_fixed_abs_candidate"] == "b"
    assert result["empirical_best_fixed_nr_candidate"] == "a"
    assert result["H_abs"] == pytest.approx(2 / 3)
    assert result["H_normalized"] == pytest.approx(1 / 3)


def test_explicit_class_seed_set_weighting_and_missing_support():
    rows = [
        dict(class_id=c, seed=s, set_id=b, policy="p", loss=float(100 * c + 10 * s + b))
        for c in range(2)
        for s in range(2)
        for b in range(3)
    ]
    values = independent_class_means(rows, [0, 1], [0, 1], [0, 1, 2], "p", "loss")
    np.testing.assert_array_equal(values, [6, 106])
    with pytest.raises(ValueError):
        independent_class_means(rows[:-1], [0, 1], [0, 1], [0, 1, 2], "p", "loss")
    with pytest.raises(ValueError):
        independent_class_means(rows + rows[:1], [0, 1], [0, 1], [0, 1, 2], "p", "loss")
    rows[0]["loss"] = None
    assert independent_class_means(rows, [0, 1], [0, 1], [0, 1, 2], "p", "loss") is None


def test_shared_indices_preserve_paired_constant_difference():
    indices = np.random.Generator(np.random.PCG64(2026091101)).integers(0, 4, (10000, 4))
    left, right = np.array([100, 3, 40, 700]), np.array([102, 5, 42, 702])
    result = independent_interval(left - right, indices)
    assert set(result.values()) == {-2.0}
    with pytest.raises(ValueError):
        independent_interval(left, indices.astype(float))


def test_inventory_hash_and_escape_failures(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(b"x\n1\n")
    row = dict(
        path=path.name,
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert validate_inventory(tmp_path, [row]) == 1
    with pytest.raises(ValueError):
        validate_inventory(tmp_path, [row, row])
    with pytest.raises(ValueError):
        validate_inventory(tmp_path, [{**row, "path": "../source.csv"}])
    path.write_bytes(b"x\n2\n")
    with pytest.raises(ValueError):
        validate_inventory(tmp_path, [row])


def test_discrete_comparison_exact_and_near_zero_floating_tolerance():
    check = Comparison()
    check.check("t", "r", "count", "3.00000000001", 3)
    check.check("t", "r", "zero", "1e-13", 0.0)
    check.check("t", "r", "flag", "False", False)
    assert len(check.failures) == 1
    assert check.failures[0]["metric"] == "count"
    check.check("t", "r", "not_a_count", False, 0)
    check.check("t", "r", "not_a_flag", 0, False)
    assert len(check.failures) == 3


@pytest.fixture(scope="module")
def independent_source_reference():
    return reference_from_sources(ROOT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("classes", [1]),
        ("seeds", [79, 80, 81, 83]),
        ("candidates", 80),
        ("primary_sets", 26),
        ("decision_units_per_policy", 1727),
        ("policy_rows", 1),
        ("scope", "prospective"),
        ("scientific_model_calls", 1),
        ("gpu_calls", False),
        ("fitted_parameters", 1),
        ("unresolved_decision_rows", 1),
        ("policies", ["candidate_B"]),
        ("unchanged_historical_decisions", 1),
    ],
)
def test_manifest_scope_tampering_rejected(independent_source_reference, field, value):
    manifest = json.loads((ROOT / "analysis/baseline_audit/audit_manifest.json").read_text())
    assert validate_baseline_contract(manifest, independent_source_reference) > 20
    manifest[field] = value
    with pytest.raises(ValueError, match="semantic mismatch"):
        validate_baseline_contract(manifest, independent_source_reference)


def test_nested_bootstrap_and_subset_tampering_rejected(independent_source_reference):
    manifest = json.loads((ROOT / "analysis/baseline_audit/audit_manifest.json").read_text())
    for section, field, value in [
        ("bootstrap", "unit", "seed"),
        ("bootstrap", "shared_paired_indices", False),
        ("exploratory_subset", "prospective", True),
    ]:
        changed = copy.deepcopy(manifest)
        changed[section][field] = value
        with pytest.raises(ValueError, match="semantic mismatch"):
            validate_baseline_contract(changed, independent_source_reference)


@pytest.mark.parametrize(
    "field,value",
    [("public_blob", "0" * 40), ("rows", 1), ("schema", ["fake"]), ("source_commit", "0" * 40)],
)
def test_source_lineage_schema_tampering_rejected(independent_source_reference, field, value):
    manifest = json.loads((ROOT / "analysis/baseline_audit/audit_manifest.json").read_text())
    records = copy.deepcopy(manifest["inputs"])
    records[0][field] = value
    with pytest.raises(ValueError):
        validate_source_metadata(ROOT, records, independent_source_reference["source_inventory"])


def test_no_main_scientific_imports():
    tree = ast.parse((ROOT / "src/rcfs_dq/baseline_independent.py").read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and name.startswith("rcfs_dq") for name in imported)


def test_public_scalar_reference_grid_and_old_results():
    reference = reference_from_sources(ROOT)
    assert len(reference["decisions"]) == 13824
    assert len(reference["classes"]) == 16 and reference["seeds"] == [79, 80, 81, 82]
    rows = {(r["policy"], r["metric"]): r for r in reference["summaries"]}
    assert rows["candidate_B", "normalized_regret"]["point"] == pytest.approx(
        0.04313539301865944, abs=1e-12
    )
    assert sum(h["fully_fixed"] for h in reference["headroom"]) == 14
    decisions = {
        (r["class_id"], r["seed"], r["set_id"], r["policy"]): r for r in reference["decisions"]
    }
    for row in reference["decisions"]:
        if row["policy"] == "pure_G":
            assert (
                row["selected"]
                == decisions[row["class_id"], row["seed"], row["set_id"], "latest_stage"][
                    "selected"
                ]
            )
    # Correct reproduction, not a requirement to win: the retrospective CI contains zero.
    delta = (
        reference["class_means"]["primary", "candidate_B", "normalized_regret"]
        - reference["class_means"]["primary", "latest_stage", "normalized_regret"]
    )
    interval = independent_interval(delta, reference["indices"])
    assert interval["low"] < 0 < interval["high"]
    assert json.dumps(reference["bootstrap_indices_sha256"])
