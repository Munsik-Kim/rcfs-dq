"""Synthetic correctness tests, not tests that a preferred policy must win."""

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import rcfs_dq.baseline_audit as baseline_module
from rcfs_dq.baseline_audit import (
    BOOTSTRAP_SEED,
    POLICIES,
    build_decisions,
    empirical_headroom,
    evaluate_policy,
    evaluate_unit,
    export_audit,
    load_inputs,
    paired_comparisons,
    read_csv,
    run_audit,
    summarize_decisions,
)
from rcfs_dq.decisions import Candidate, class_block_ci
from rcfs_dq.verification import file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def members():
    return tuple(Candidate(name, "g", stage, 4, 8) for name, stage in zip("abc", (4, 10, 15)))


@pytest.fixture
def frozen():
    return {"G": {"a": 3.0, "b": 2.0, "c": 1.0}, "beta": 1.0, "constants": {"set": "a"}}


@pytest.fixture(scope="module")
def actual_data():
    return load_inputs(ROOT)


def test_latest_uses_stage_index_and_source_tie_order(members, frozen):
    energies = {"a": 1.0, "b": 1.0, "c": 1.0}
    for policy in ("latest_stage", "stage_position_only"):
        assert evaluate_policy(members[::-1], energies, frozen, "set", policy) == "c"
    tied = (Candidate("late_z", "z", 15, 4, 8, 1), Candidate("late_a", "a", 15, 4, 8, 0))
    assert (
        evaluate_policy(tied, {c.cell_id: 1.0 for c in tied}, frozen, "set", "latest_stage")
        == "late_a"
    )


def test_pure_G_alias_is_data_dependent(members, frozen):
    energies = dict.fromkeys("abc", 1.0)
    assert evaluate_policy(members, energies, frozen, "set", "pure_G") == "c"
    changed = copy.deepcopy(frozen)
    changed["G"]["a"] = 0.01
    assert evaluate_policy(members, energies, changed, "set", "pure_G") == "a"
    assert evaluate_policy(members, energies, changed, "set", "latest_stage") == "c"


def test_identical_G_argmin_does_not_fix_energy_geometry_tradeoff(members, frozen):
    energies = {"a": 1.0, "b": 1.0, "c": 100.0}
    f1 = dict(frozen, G={"a": 10.0, "b": 5.0, "c": 1.0})
    f2 = dict(frozen, G={"a": 1000.0, "b": 500.0, "c": 1.0})
    assert evaluate_policy(members, energies, f1, "set", "pure_G") == "c"
    assert evaluate_policy(members, energies, f2, "set", "pure_G") == "c"
    assert evaluate_policy(members, energies, f1, "set", "candidate_B") == "b"
    assert evaluate_policy(members, energies, f2, "set", "candidate_B") == "c"


@pytest.mark.parametrize(
    "beta,geometry", [(0, {"a": 3.0, "b": 2.0, "c": 1.0}), (1, dict.fromkeys("abc", 2.0))]
)
def test_beta_zero_or_equal_G_preserves_energy_choices(members, frozen, beta, geometry):
    energy = {"a": 7.0, "b": 2.0, "c": 3.0}
    config = dict(frozen, beta=beta, G=geometry)
    assert evaluate_policy(members, energy, config, "set", "candidate_B") == "b"
    assert evaluate_policy(members, energy, config, "set", "energy") == "b"


def test_hybrid_low_risk_orientation_protects_M3(frozen):
    members = (Candidate("a", "m", 4, 4, 8, regime="M3"), Candidate("b", "l", 10, 4, 8))
    energy = {"a": 0.0, "b": 100.0}
    assert evaluate_policy(members, energy, frozen, "set", "candidate_B") == "a"
    assert evaluate_policy(members, energy, frozen, "set", "hybrid") == "b"


@pytest.mark.parametrize("selected", ["a", "b", "c", None])
def test_oracle_regret_and_uniform_expected(members, selected):
    result = evaluate_unit(members, {"a": 1.0, "b": 2.0, "c": 7.0}, selected)
    assert result["oracle"] == "a"
    expected = 7 / 3 if selected is None else {"a": 0.0, "b": 1.0, "c": 6.0}[selected]
    assert result["absolute_regret"] == pytest.approx(expected)
    assert result["normalized_regret"] == pytest.approx(expected / 6)
    assert result["absolute_regret"] >= 0


def test_uniform_canonical_hit_and_all_minimum_ties_are_distinct(members):
    result = evaluate_unit(members, {"a": 1.0, "b": 1.0, "c": 4.0}, None)
    assert result["oracle"] == "a"
    assert result["oracle_hit"] == 1 / 3
    assert result["all_minima_hit"] == 2 / 3


@pytest.mark.parametrize("value", [0.0, 0.1, 1.0, 1e200])
@pytest.mark.parametrize("selected", ["c", None])
def test_zero_range_preserves_zero_absolute_and_undefined_normalized(members, value, selected):
    result = evaluate_unit(members, dict.fromkeys("abc", value), selected)
    assert not result["resolved"]
    assert result["normalized_regret"] is None
    assert result["absolute_regret"] == 0


def test_tiny_range_not_arbitrary_epsilon_normalized(members):
    result = evaluate_unit(members, {"a": 1.0, "b": 1.0 + 1e-13, "c": 1.0}, "b")
    assert result["absolute_regret"] > 0
    assert result["normalized_regret"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("selected", ["a", None])
def test_invalid_risk_rejected(members, bad, selected):
    with pytest.raises(ValueError):
        evaluate_unit(members, {"a": bad, "b": 1.0, "c": 2.0}, selected)


def test_missing_duplicate_and_cost_mismatch_rejected(members):
    with pytest.raises(ValueError):
        evaluate_unit(members, {"a": 1.0, "b": 2.0}, "a")
    with pytest.raises(ValueError):
        evaluate_unit((members[0], members[0]), {"a": 1.0}, "a")
    wrong_cost = (members[0], Candidate("z", "g", 15, 4, 9))
    with pytest.raises(ValueError):
        evaluate_unit(wrong_cost, {"a": 1.0, "z": 2.0}, "a")


def test_constant_never_reselected_from_Y(members, frozen):
    energies = dict.fromkeys("abc", 1.0)
    choice = evaluate_policy(members, energies, frozen, "set", "constant")
    assert choice == "a"
    assert (
        evaluate_unit(members, {"a": 100.0, "b": 0.0, "c": 1.0}, choice)["absolute_regret"] == 100.0
    )
    assert evaluate_policy(members, energies, frozen, "set", "constant") == "a"


def test_sealed_choices_have_no_test_Y_input(actual_data):
    original = build_decisions(actual_data)
    mutated = copy.deepcopy(actual_data)
    for records in mutated["observations"].values():
        for row in records.values():
            row["native_Y"] = 1.0
    changed = build_decisions(mutated)
    assert [r["selected"] for r in original] == [r["selected"] for r in changed]
    assert all(r["normalized_regret"] is None for r in changed)


def synthetic_decisions():
    rows = []
    for c in (1, 2):
        for seed in (10, 20):
            for cost_set in ("x", "y"):
                value = c * 100 + seed + (1 if cost_set == "x" else 9)
                for policy, shift in (("a", 0), ("b", 10)):
                    rows.append(
                        dict(
                            family="primary",
                            policy=policy,
                            class_id=c,
                            seed=seed,
                            set_id=cost_set,
                            absolute_regret=float(value + shift),
                            normalized_regret=float(value + shift),
                            oracle_hit=0.0,
                        )
                    )
    return rows


def test_equal_set_seed_class_aggregation_and_shared_pairing():
    rows = synthetic_decisions()
    indices = np.random.Generator(np.random.PCG64(12)).integers(0, 2, (100, 2))
    _, classes, means = summarize_decisions(rows, indices)
    np.testing.assert_array_equal(means["primary", "a", "absolute_regret"], [120.0, 220.0])
    pairs, _, _ = paired_comparisons(means, indices, (("a", "b"),))
    result = next(r for r in pairs if r["metric"] == "absolute_regret")
    assert all(result[field] == -10 for field in ("point", "low", "high", "lower95", "upper95"))
    assert len(classes) == 12


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "undefined"])
def test_no_silent_aggregation_support_change(mutation):
    rows = synthetic_decisions()
    indices = np.array([[0, 1], [1, 0]])
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["normalized_regret"] = None
        summary, _, means = summarize_decisions(rows, indices)
        r = next(r for r in summary if r["policy"] == "a" and r["metric"] == "normalized_regret")
        assert r["point"] is None and r["undefined_units"] == 1
        assert ("primary", "a", "normalized_regret") not in means
        return
    with pytest.raises(ValueError):
        summarize_decisions(rows, indices)


def test_best_fixed_not_most_frequent_oracle_and_objectives_differ():
    result = empirical_headroom([[0.0, 1.0], [0.0, 1.0], [100.0, 0.0]], ["a", "b"])
    assert result["majority_candidate"] == "a"
    assert result["empirical_best_fixed_abs_candidate"] == "b"
    assert result["empirical_best_fixed_nr_candidate"] == "a"
    assert result["H_abs"] == pytest.approx(2 / 3)
    assert result["H_normalized"] == pytest.approx(1 / 3)
    assert not result["majority_equals_minimum_expected_risk"]


def test_headroom_zero_range_cannot_claim_normalized_optimum():
    result = empirical_headroom([[1.0, 1.0], [2.0, 3.0]], ["a", "b"])
    assert result["H_abs"] == 0
    assert result["H_normalized"] is None
    assert result["headroom_normalized_valid_units"] == 1


def test_source_hash_change_is_rejected_before_analysis(tmp_path):
    (tmp_path / "PUBLIC_EVIDENCE_MANIFEST.json").write_text(
        json.dumps({"files": [{"path": "one.csv", "bytes": 1, "sha256": "0" * 64}]})
    )
    (tmp_path / "one.csv").write_text("x")
    with pytest.raises(AssertionError, match="Inventory mismatch"):
        load_inputs(tmp_path)


@pytest.fixture
def copied_source(tmp_path):
    manifest = json.loads((ROOT / "PUBLIC_EVIDENCE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], target)
    shutil.copyfile(
        ROOT / "PUBLIC_EVIDENCE_MANIFEST.json", tmp_path / "PUBLIC_EVIDENCE_MANIFEST.json"
    )
    return tmp_path


def update_fixture_manifest(root, name):
    path = root / "PUBLIC_EVIDENCE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        if row["path"] == name:
            row.update(bytes=(root / name).stat().st_size, sha256=file_sha256(root / name))
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "negative", "score"])
def test_coverage_or_numeric_source_mutation_rejected(copied_source, mutation):
    name = "evidence/e1pred2r2/candidate_observations.csv"
    path = copied_source / name
    rows = read_csv(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        field = "source_B_score" if mutation == "score" else "native_Y"
        rows[0][field] = {"nonfinite": "nan", "negative": "-1", "score": "123"}[mutation]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    update_fixture_manifest(copied_source, name)
    with pytest.raises((ValueError, AssertionError)):
        load_inputs(copied_source)


def test_original_source_and_new_retrospective_outputs_are_correct(actual_data):
    tables, manifest = run_audit(ROOT)
    assert manifest["decision_units_per_policy"] == 1728
    # 12 original primary summary rows + eight original paired rows, five CI fields each.
    assert manifest["source_summary_numeric_checks"] == 100
    assert manifest["scientific_model_calls"] == 0
    assert manifest["bootstrap"]["seed"] == BOOTSTRAP_SEED
    assert len(tables["decisions"]) == 1728 * len(POLICIES)
    assert len(actual_data["classes"]) == 16 and len(actual_data["seeds"]) == 4
    # Correct results include harm in the absolute-risk objective; do not gate a preferred winner.
    row = next(
        r
        for r in tables["paired_comparisons"]
        if (r["left"], r["right"], r["metric"])
        == ("candidate_B", "latest_stage", "absolute_regret")
    )
    assert row["point"] == pytest.approx(0.16660797222310794)
    assert row["low"] == pytest.approx(0.024958945016602403)
    assert all(r["analysis_status"] == "source_summary_only" for r in tables["secondary_results"])


def test_no_torch_model_import_from_scalar_audit():
    code = "import sys; import rcfs_dq.baseline_audit; assert 'torch' not in sys.modules"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_forbidden_output_and_symlink_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        export_audit(ROOT, ROOT / "evidence")
    with pytest.raises(ValueError):
        export_audit(ROOT, ROOT)
    symlink = tmp_path / "redirect"
    symlink.symlink_to(ROOT / "evidence", target_is_directory=True)
    with pytest.raises(ValueError):
        export_audit(ROOT, symlink)


def test_bootstrap_quantile_linear_matches_explicit_paired_draws():
    values = np.array([1.0, 2.0, 10.0])
    indices = np.array([[0, 0, 2], [1, 1, 1], [0, 1, 2], [2, 2, 2]])
    result = class_block_ci(values, indices=indices)
    expected = np.quantile(
        values[indices].mean(axis=1), [0.025, 0.975, 0.05, 0.95], method="linear"
    )
    np.testing.assert_array_equal(
        [result[k] for k in ("low", "high", "lower95", "upper95")], expected
    )


@pytest.fixture
def tiny_export_root(tmp_path, monkeypatch):
    """Exercise exporter guards without copying or changing any scientific input."""
    root = tmp_path / "repository"
    for name in ("src/rcfs_dq/baseline_audit.py", "scripts/audit_baselines.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic exporter fixture\n")
    tables = {"first": [{"value": 1.0}], "last": [{"value": 2.0}]}
    manifest = dict(
        status="COMPLETED_PUBLIC_SCALAR_AUDIT",
        scope=baseline_module.SCOPE,
        original_public_manifest_sha256="1" * 64,
        source_base_commit="2" * 40,
        classes=[1, 2],
        policy_rows=2,
        scientific_model_calls=0,
        limits=["synthetic exporter fixture, no scientific observations"],
    )
    monkeypatch.setattr(
        baseline_module,
        "run_audit",
        lambda _: (copy.deepcopy(tables), copy.deepcopy(manifest)),
    )
    return root, tables, manifest


@pytest.mark.parametrize(
    "field,value",
    [
        ("classes", [99]),
        ("scope", "wrong scope"),
        ("source_base_commit", "0" * 40),
        ("scientific_model_calls", 123),
        ("limits", []),
    ],
)
def test_readonly_export_check_covers_entire_manifest(tiny_export_root, tmp_path, field, value):
    root, _, _ = tiny_export_root
    out = tmp_path / "external"
    export_audit(root, out)
    path = out / "audit_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(AssertionError, match="audit_manifest.json"):
        export_audit(root, out, check=True)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_external_nonempty_output_cannot_be_overwritten(tiny_export_root, tmp_path):
    root, _, _ = tiny_export_root
    out = tmp_path / "external"
    out.mkdir()
    protected = out / "first.csv"
    protected.write_text("user-owned content\n")
    with pytest.raises(ValueError, match="new or empty"):
        export_audit(root, out)
    assert protected.read_text() == "user-owned content\n"
    assert not (out / "audit_manifest.json").exists()


def test_canonical_regeneration_requires_prior_output_inventory(tiny_export_root):
    root, tables, _ = tiny_export_root
    out = root / "analysis/baseline_audit"
    export_audit(root, out)
    tables["first"][0]["value"] = 3.0
    # An intentional code-analysis change may replace intact, owned outputs.
    export_audit(root, out)
    assert "3.0" in (out / "first.csv").read_text()
    (out / "last.csv").write_text("user alteration\n")
    tables["first"][0]["value"] = 4.0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match="Modified canonical"):
        export_audit(root, out)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize("symlink_name", ["last.csv", "audit_manifest.json", "unrelated_link"])
def test_all_output_symlinks_rejected_before_any_write(tiny_export_root, tmp_path, symlink_name):
    root, tables, _ = tiny_export_root
    out = root / "analysis/baseline_audit"
    export_audit(root, out)
    target = tmp_path / "victim"
    target.write_text("unchanged\n")
    link = out / symlink_name
    if link.exists():
        link.unlink()
    link.symlink_to(target)
    tables["first"][0]["value"] = 5.0
    before = (out / "first.csv").read_bytes()
    with pytest.raises(ValueError, match="Symlink"):
        export_audit(root, out)
    assert (out / "first.csv").read_bytes() == before
    assert target.read_text() == "unchanged\n"


def test_random_expected_frequency_uses_version_stable_accumulation(actual_data):
    tables, _ = run_audit(ROOT)
    rows = [
        row
        for row in tables["policy_selection_frequency"]
        if row["policy"] == "uniform_random_expected"
    ]
    assert len(rows) == 81
    assert all(row["count"] == 64.0 / 3.0 and row["frequency"] == 1.0 / 3.0 for row in rows)
