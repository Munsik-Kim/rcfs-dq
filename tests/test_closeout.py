"""Scalar arithmetic and claim boundaries, without private raw or models."""

import ast
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rcfs_dq.closeout_verification import Checks, average, verify_bridge, verify_closeout

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return verify_closeout(ROOT)


def test_public_reference_scope_and_counts(result):
    assert result["passed"]
    assert result["model_calls"] == result["GPU_calls"] == 0
    assert result["BRIDGE0"]["available"] == 192
    assert result["BRIDGE0"]["missing"] == 96
    assert result["BRIDGE0"]["three_probe_summary"] is None
    assert result["BRIDGE0"]["help_hurt_tie"] == [178, 14, 0]
    point, lower, upper = result["BRIDGE0"]["Full_minus_Scalar"]
    assert lower < 0 < upper and point > 0
    assert result["V3"]["C_failures"] == 1 and result["V3"]["D_failures"] == 13
    assert result["V3"]["C_by_family"]["accumulated"] is False


def test_independent_does_not_import_research_algorithms():
    tree = ast.parse((ROOT / "src/rcfs_dq/closeout_verification.py").read_text())
    assert not any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(x and ("j0_" in x or "torch" in x) for x in imports)


def test_optimized_public_verification_is_identical():
    outputs = []
    for options in ([], ["-O"]):
        proc = subprocess.run(
            [sys.executable, *options, "scripts/verify_closeout_evidence.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1]


def test_paired_median_is_not_policy_median_difference():
    left, right = np.array([1.0, 4.0, 8.0, 20.0]), np.array([0.0, 2.0, 15.0, 17.0])
    assert average(list(left - right), median=True) != np.median(left) - np.median(right)
    assert average([1.0, None]) is None


@pytest.mark.parametrize("column", ["err_sq_full", "abs_full_minus_scalar"])
def test_scalar_tamper_fails_without_using_manifest_as_answer(tmp_path, column):
    dest = tmp_path / "analysis/closeout/bridge0"
    shutil.copytree(ROOT / "analysis/closeout/bridge0", dest)
    path = dest / "natural_error_rows.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, data = reader.fieldnames, list(reader)
    data[0][column] = str(float(data[0][column]) + 0.1)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    with pytest.raises(ValueError, match="Scalar mismatch"):
        verify_bridge(tmp_path, Checks())


def test_public_closeout_does_not_replace_historical_gate():
    status = json.loads((ROOT / "analysis/closeout/PROJECT_STATUS.json").read_text())
    assert status["historical_scientific_statuses"]["J0_v1"] == "J0_NUMERICAL_METHOD_REVIEW"
    assert status["project_level_closeout"] == "RCFS_DQ_LOCAL_MECHANISM_TRACK_CLOSED_AS_LIMITED"
    assert status["deferred_is_not_disproven"] and not status["new_scientific_execution_authorized"]


@pytest.mark.parametrize("document", ["RESEARCH_SUMMARY.md", "LIMITATIONS.md", "RESULTS.md"])
def test_new_document_boundaries_and_links(document):
    path = ROOT / "docs" / document
    content = path.read_text()
    assert "probe16" in content and "Scalar" in content
    for target in re.findall(r"\]\(([^)]+)\)", content):
        if "://" not in target:
            assert (path.parent / target).is_file()


def test_exact_primary_numbers_remain_in_canonical_results(result):
    text = (ROOT / "docs/RESULTS.md").read_text().replace("−", "-")
    for value in result["BRIDGE0"]["Full_minus_Scalar"]:
        assert str(value) in text
    assert "178/192" in text and "14/192" in text
    assert "Neither superiority nor equivalence" in text


def test_ci_requires_public_reference_and_keeps_other_checks():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "python scripts/verify_closeout_evidence.py" in ci
    assert "python -O scripts/verify_closeout_evidence.py" in ci
    assert "PUBLIC_CLOSEOUT_MANIFEST.json" in ci
