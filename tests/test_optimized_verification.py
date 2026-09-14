"""Normal and optimized Python must enforce the same scientific checks.

All corruption is confined to temporary copies. Test-only manifest resealing
isolates the semantic checks behind the immutable production inventory gate.
"""

import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence_copy(tmp_path):
    for directory in ("configs", "evidence", "figures", "scripts"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    shutil.copy2(ROOT / "PUBLIC_EVIDENCE_MANIFEST.json", tmp_path)
    return tmp_path


def run_verifier(root, optimized, *, resealed_test_manifest=False):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    script = root / "scripts/verify_evidence.py"
    if resealed_test_manifest:
        # Deliberately trust a TEST fixture inventory to reach the independent
        # semantic guard. Production commands never permit a trust-root override.
        harness = (
            "import hashlib,json,runpy,sys; "
            "from pathlib import Path; "
            "v=runpy.run_path(sys.argv[1]); "
            "g=v['verify'].__globals__; "
            "g['ORIGINAL_MANIFEST_SHA256']=hashlib.sha256("
            "(Path(sys.argv[1]).parents[1]/'PUBLIC_EVIDENCE_MANIFEST.json').read_bytes())"
            ".hexdigest(); print(json.dumps(v['verify'](),sort_keys=True))"
        )
        command.extend(["-c", harness, str(script)])
    else:
        command.append(str(script))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["HF_HUB_OFFLINE"] = "1"
    return subprocess.run(command, capture_output=True, text=True, env=environment, check=False)


def edit_csv(path, edit):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        data = list(reader)
    edit(data)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def reseal_test_inventory(root):
    path = root / "PUBLIC_EVIDENCE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        payload = (root / row["path"]).read_bytes()
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    path.write_text(json.dumps(manifest))


def test_valid_normal_and_optimized_json_identical(evidence_copy):
    normal = run_verifier(evidence_copy, False)
    optimized = run_verifier(evidence_copy, True)
    assert normal.returncode == optimized.returncode == 0
    assert json.loads(normal.stdout) == json.loads(optimized.stdout)
    result = json.loads(normal.stdout)
    assert result["scalar_comparisons"] == 19192
    assert result["confirmation_signs"] == 9
    assert result["confirmation_stricter_replications"] == 8
    assert result["model_calls"] == 0


@pytest.mark.parametrize("corruption", ["manifest", "hash", "missing"])
def test_original_inventory_corruption_rejected_with_and_without_optimization(
    evidence_copy, corruption
):
    manifest_path = evidence_copy / "PUBLIC_EVIDENCE_MANIFEST.json"
    payload = evidence_copy / "configs/candidates.csv"
    if corruption == "manifest":
        manifest_path.write_text('{"files": []}')
    elif corruption == "hash":
        payload.write_bytes(payload.read_bytes() + b"\n")
    else:
        payload.unlink()
    results = [run_verifier(evidence_copy, optimized) for optimized in (False, True)]
    assert all(result.returncode != 0 and not result.stdout for result in results)
    assert results[0].stderr.splitlines()[-1] == results[1].stderr.splitlines()[-1]


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("packed_cost", "Candidate theoretical packed cost mismatch"),
        ("shape", "Shape/numel mismatch"),
        ("duplicate", "Duplicate candidate observation"),
        ("coverage", "Native observation coverage mismatch"),
        ("native_risk", "Numerical mismatch"),
        ("discovery_status", "Discovery tangent status mismatch"),
        ("confirmation_sign", "Confirmation sign count mismatch"),
        ("confirmation_strict", "Confirmation strict replication count mismatch"),
        ("tail_decision", "Tail0 historical decision mismatch"),
        ("tail_confusion", "Tail0 confusion matrix mismatch"),
    ],
)
def test_semantic_corruption_rejected_with_and_without_optimization(
    evidence_copy, corruption, expected
):
    if corruption in {"packed_cost", "shape"}:

        def mutate(data):
            if corruption == "packed_cost":
                data[0]["packed_bits"] = str(int(data[0]["packed_bits"]) + 1)
            else:
                shapes = json.loads(data[0]["shapes"])
                shapes[next(iter(shapes))][0] += 1
                data[0]["shapes"] = json.dumps(shapes)

        edit_csv(evidence_copy / "configs/candidates.csv", mutate)
    elif corruption in {"duplicate", "coverage", "native_risk"}:

        def mutate(data):
            if corruption == "duplicate":
                data.append(data[0].copy())
            elif corruption == "coverage":
                data.pop()
            else:
                for row in data:
                    row["native_Y"] = str(float(row["native_Y"]) * 2)

        edit_csv(evidence_copy / "evidence/e1pred2r2/candidate_observations.csv", mutate)
    elif corruption == "discovery_status":

        def mutate(data):
            row = next(row for row in data if row["alpha_role"] == "fixed_alpha")
            row["cell_tangent_status"] = "INVALID_TEST_FIXTURE"

        edit_csv(evidence_copy / "evidence/e1d1/directional_contrasts.csv", mutate)
    elif corruption.startswith("confirmation"):

        def mutate(data):
            # Mutate the copied table; the verifier selects the nine positive
            # sentinels independently from the locked global-test membership.
            key = "same_sign" if corruption == "confirmation_sign" else "replicated"
            for row in data:
                row[key] = "False"

        edit_csv(evidence_copy / "evidence/e1c/directional_replication.csv", mutate)
    else:
        path = evidence_copy / "evidence/e1tail0/analysis_summary.json"
        data = json.loads(path.read_text())
        if corruption == "tail_decision":
            data["provisional_decision"] = "INVALID_TEST_FIXTURE"
        else:
            data["detection"]["TP"] += 1
        path.write_text(json.dumps(data))
    reseal_test_inventory(evidence_copy)
    results = [
        run_verifier(evidence_copy, optimized, resealed_test_manifest=True)
        for optimized in (False, True)
    ]
    assert all(result.returncode != 0 and expected in result.stderr for result in results)
    assert results[0].stderr.splitlines()[-1] == results[1].stderr.splitlines()[-1]


def test_production_scalar_verification_has_no_removable_assertions():
    for name in (
        "scripts/verify_evidence.py",
        "src/rcfs_dq/verification.py",
        "src/rcfs_dq/decisions.py",
        "src/rcfs_dq/scoring.py",
        "src/rcfs_dq/geometry.py",
    ):
        tree = ast.parse((ROOT / name).read_text())
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name
