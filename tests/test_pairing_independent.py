"""Independent P1 algebra and scalar readback tests, with no model execution."""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rcfs_dq.pairing_independent import (
    independent_stage_coefficients,
    pairing_reference,
    validate_pairing_contract,
    verify_pairing_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage_coefficient_equal_cell_weight_not_group_weight():
    candidates = {"a": {"stage": 4}, "b": {"stage": 4}, "c": {"stage": 4}, "d": {"stage": 15}}
    gains = dict(a=1.0, b=4.0, c=16.0, d=9.0)
    rows = independent_stage_coefficients(candidates, gains, 0.5)
    assert rows[0]["source_cell_count"] == 3
    assert rows[0]["coefficient"] == pytest.approx(2)
    assert rows[1]["coefficient"] == pytest.approx(3)
    reversed_rows = independent_stage_coefficients(
        dict(reversed(list(candidates.items()))), gains, 0.5
    )
    assert rows == reversed_rows


def test_beta_zero_and_equal_geometry_identities():
    candidates = {"a": {"stage": 4}, "b": {"stage": 15}}
    assert all(
        row["coefficient"] == 1
        for row in independent_stage_coefficients(candidates, dict(a=2, b=20), 0)
    )
    rows = independent_stage_coefficients(candidates, dict(a=5, b=5), 0.3)
    assert rows[0]["coefficient"] == rows[1]["coefficient"]


@pytest.mark.parametrize(
    "gains,beta",
    [(dict(a=1), 1), (dict(a=1, b=0), 1), (dict(a=1, b=np.inf), 1), (dict(a=1, b=2), np.nan)],
)
def test_invalid_or_missing_geometry_refused(gains, beta):
    with pytest.raises(ValueError):
        independent_stage_coefficients({"a": {"stage": 4}, "b": {"stage": 15}}, gains, beta)


def test_coefficient_function_has_no_outcome_input_and_no_main_imports():
    assert list(inspect.signature(independent_stage_coefficients).parameters) == [
        "candidates",
        "gains",
        "beta",
    ]
    tree = ast.parse((ROOT / "src/rcfs_dq/pairing_independent.py").read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(
        name in {"pairing_audit", "baseline_audit", "decisions", "scoring"} for name in imported
    )


def test_full_scalar_reference_support_and_stage_vs_detailed_choices():
    result = pairing_reference(ROOT)
    assert len(result["decisions"]) == 13824
    assert len(result["predictor_scores"]) == 25920
    assert all(row["source_cell_count"] == 27 for row in result["stage_coefficients"])
    lookup = {(r["class_id"], r["seed"], r["set_id"], r["policy"]): r for r in result["decisions"]}
    changed = 0
    for row in result["decisions"]:
        if row["policy"] == "candidate_B":
            other = lookup[row["class_id"], row["seed"], row["set_id"], "stage_only"]
            changed += row["selected"] != other["selected"]
    assert changed == 78


def test_main_and_independent_scalar_readback():
    report, rows = verify_pairing_audit(ROOT, ROOT / "analysis/pairing_ablation")
    assert report["status"] == "PASS"
    assert report["model_calls"] == 0
    assert report["source_alignment_independently_reexecuted"] is False
    assert report["original_receiver_tensors_independently_recomputed"] is False
    assert rows and all(row["passed"] for row in rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("retrospective", False),
        ("class_count", 15),
        ("seed_count", 3),
        ("model_calls", 1),
        ("source_coefficients_modified", True),
    ],
)
def test_p1_manifest_semantic_tampering_rejected(field, value):
    manifest = json.loads((ROOT / "analysis/pairing_ablation/audit_manifest.json").read_text())
    reference = pairing_reference(ROOT)
    assert validate_pairing_contract(manifest, reference) > 10
    manifest[field] = value
    with pytest.raises(ValueError, match="semantic mismatch"):
        validate_pairing_contract(manifest, reference)
