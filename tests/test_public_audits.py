"""Optimization-independent verification and separate derived-layer invariants."""

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rcfs_dq.public_audits import (
    IMPLEMENTATION,
    MANIFEST_NAME,
    AuditVerificationError,
    _path,
    manifest_from_files,
    verify_public_audits,
)

ROOT = Path(__file__).resolve().parents[1]


def test_separate_manifest_and_original_layer():
    result = manifest_from_files(ROOT)
    saved = json.loads((ROOT / MANIFEST_NAME).read_text())
    assert saved == result
    assert result["original_evidence_files_checked"] == 22
    names = {r["path"] for r in result["artifacts"]}
    assert names and all(name.startswith("analysis/") for name in names)
    assert MANIFEST_NAME not in names
    assert all(r["derived"] is True and r["model_calls"] == 0 for r in result["artifacts"])
    assert all(r["source_inputs"] and r["analysis_script"] for r in result["artifacts"])
    assert all(
        result[key] == 0
        for key in (
            "model_calls",
            "gpu_calls",
            "new_model_observations",
            "predictor_refits",
            "new_prospective_confirmation",
        )
    )


def test_committed_new_audit_csvs_use_lf_only():
    for directory in ("baseline_audit", "pairing_ablation"):
        files = list((ROOT / "analysis" / directory).glob("*.csv"))
        assert files
        for path in files:
            payload = path.read_bytes()
            assert b"\r" not in payload, path.name
            assert payload.endswith(b"\n"), path.name


def test_main_audit_csv_serializers_explicitly_use_lf():
    from rcfs_dq.baseline_audit import csv_bytes as baseline_csv
    from rcfs_dq.pairing_audit import csv_bytes as pairing_csv

    rows = [{"metric": "a", "value": 1.25}, {"metric": "b", "value": -0.5}]
    expected = b"metric,value\na,1.25\nb,-0.5\n"
    assert baseline_csv(rows) == pairing_csv(rows) == expected


def test_independent_public_audit_complete():
    result = verify_public_audits(ROOT)
    assert result["passed"] and result["model_calls"] == 0
    assert result["p0_independent_scalar_fields"] > 230000
    assert result["p1_independent_scalar_fields"] > 228000


@pytest.mark.parametrize("name", ["", "../escape", "/absolute", "a/../b", "a\\b", "a//b"])
def test_public_audit_path_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(AuditVerificationError):
        _path(tmp_path, name)


def test_public_audit_path_rejects_symlink(tmp_path):
    (tmp_path / "target").write_text("x")
    (tmp_path / "alias").symlink_to(tmp_path / "target")
    with pytest.raises(AuditVerificationError):
        _path(tmp_path, "alias")


@pytest.fixture(scope="module")
def public_scalar_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("scalar-public-copy")
    manifest = json.loads((ROOT / MANIFEST_NAME).read_text())
    original = json.loads((ROOT / "PUBLIC_EVIDENCE_MANIFEST.json").read_text())
    names = {MANIFEST_NAME, "PUBLIC_EVIDENCE_MANIFEST.json"}
    names.update(r["path"] for r in original["files"])
    names.update(r["path"] for r in manifest["artifacts"])
    names.update(IMPLEMENTATION)
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return root


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "scope",
        "zero_calls",
        "omitted_artifact",
        "altered_hash",
        "bool_calls",
        "int_derived",
        "float_bytes",
    ],
)
def test_invalid_manifest_fails_with_and_without_optimization(
    public_scalar_copy, optimize, mutation
):
    root = public_scalar_copy
    original = (ROOT / MANIFEST_NAME).read_text()
    manifest = json.loads(original)
    changed = copy.deepcopy(manifest)
    if mutation == "scope":
        changed["scope"] = "prospective confirmation"
    elif mutation == "zero_calls":
        changed["model_calls"] = 1
    elif mutation == "omitted_artifact":
        changed["artifacts"].pop()
    elif mutation == "bool_calls":
        changed["model_calls"] = False
    elif mutation == "int_derived":
        changed["derived"] = 1
    elif mutation == "float_bytes":
        changed["artifacts"][0]["bytes"] = float(changed["artifacts"][0]["bytes"])
    else:
        changed["artifacts"][0]["sha256"] = "0" * 64
    (root / MANIFEST_NAME).write_text(json.dumps(changed))
    try:
        command = [
            sys.executable,
            *(["-O"] if optimize else []),
            str(ROOT / "scripts/verify_public_audits.py"),
            "--root",
            str(root),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        )
        assert result.returncode == 1
        assert json.loads(result.stdout)["passed"] is False
        assert str(root) not in result.stdout
        assert not result.stderr
    finally:
        (root / MANIFEST_NAME).write_text(original)


def test_new_production_audits_have_no_optimization_sensitive_asserts():
    for name in IMPLEMENTATION:
        if name in {
            "src/rcfs_dq/verification.py",
            "src/rcfs_dq/decisions.py",
            "src/rcfs_dq/scoring.py",
        }:
            continue
        tree = ast.parse((ROOT / name).read_text())
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name


def test_aggregate_verifier_does_not_import_main_scoring_or_audit():
    tree = ast.parse((ROOT / "src/rcfs_dq/public_audits.py").read_text())
    forbidden = {"baseline_audit", "pairing_audit", "scoring", "decisions"}
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden for node in ast.walk(tree)
    )


def test_manifest_builder_does_not_overwrite_changed_manifest(public_scalar_copy):
    root = public_scalar_copy
    path = root / MANIFEST_NAME
    original = path.read_text()
    path.write_text("{}\n")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_public_audit_manifest.py"),
                "--root",
                str(root),
            ],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        )
        assert result.returncode != 0
        assert path.read_text() == "{}\n"
    finally:
        path.write_text(original)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("name", ["independent_verification.json", "main_vs_independent.csv"])
def test_independent_cli_preserves_existing_external_outputs(tmp_path, optimize, name):
    audit = tmp_path / "external_audit"
    audit.mkdir()
    target = audit / name
    target.write_text("user-owned content\n")
    command = [
        sys.executable,
        *(["-O"] if optimize else []),
        str(ROOT / "scripts/verify_baseline_audit.py"),
        "--audit",
        str(audit),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    )
    assert result.returncode != 0
    assert "External verification outputs must be new" in result.stderr
    assert target.read_text() == "user-owned content\n"


@pytest.mark.parametrize("name", ["independent_verification.json", "main_vs_independent.csv"])
def test_independent_cli_preserves_unrecognized_canonical_outputs(public_scalar_copy, name):
    root = public_scalar_copy
    target = root / "analysis/baseline_audit" / name
    original = target.read_bytes()
    target.write_text("{}\n")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_baseline_audit.py"), "--root", str(root)],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        )
        assert result.returncode != 0
        assert target.read_text() == "{}\n"
    finally:
        target.write_bytes(original)
