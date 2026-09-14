"""Separate provenance and independent checks for published P0/P1 scalar audits.

This module never imports the main score, decision, or audit implementation.
Manifest construction hashes existing derived files; verification separately
recalculates decisions and paired class-block intervals from original scalars.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

from .baseline_independent import validate_inventory, verify_baseline_audit
from .pairing_independent import verify_pairing_audit

MANIFEST_NAME = "PUBLIC_AUDIT_MANIFEST.json"
ORIGINAL_MANIFEST_SHA256 = "8a5ffc01d6f673fcd46c4c9f404baadc4a76fa15e014c149740c00ce44e946ba"
CERTIFICATE_SHA256 = "f18b68a489ec71bc4194cfbac5d361aba26c8b2f1d2fe871706e51122068d6d8"
AUDITS = (
    ("P0", "analysis/baseline_audit", "scripts/audit_baselines.py"),
    ("P1", "analysis/pairing_ablation", "scripts/audit_pairing_ablation.py"),
)
IMPLEMENTATION = (
    "src/rcfs_dq/baseline_audit.py",
    "src/rcfs_dq/baseline_independent.py",
    "src/rcfs_dq/pairing_audit.py",
    "src/rcfs_dq/pairing_independent.py",
    "src/rcfs_dq/public_audits.py",
    "src/rcfs_dq/decisions.py",
    "src/rcfs_dq/scoring.py",
    "src/rcfs_dq/verification.py",
    "scripts/audit_baselines.py",
    "scripts/audit_pairing_ablation.py",
    "scripts/verify_baseline_audit.py",
    "scripts/verify_public_audits.py",
    "scripts/build_public_audit_manifest.py",
)


class AuditVerificationError(ValueError):
    """An explicit, optimization-independent audit integrity failure."""


def _path(root, name):
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or str(relative) != name
    ):
        raise AuditVerificationError("Unsafe public audit path")
    path = root / name
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise AuditVerificationError("Symlink public audit input")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise AuditVerificationError("Non-regular or missing public audit input")
    return path


def _record(root, name):
    path = _path(root, name)
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def manifest_from_files(root):
    """Deterministic inventory, excluding its own file to avoid a self-hash cycle."""
    root = Path(root).resolve()
    original = _record(root, "PUBLIC_EVIDENCE_MANIFEST.json")
    if original["sha256"] != ORIGINAL_MANIFEST_SHA256:
        raise AuditVerificationError("Original scientific evidence manifest changed")
    original_records = json.loads(_path(root, original["path"]).read_text())["files"]
    validate_inventory(root, original_records)
    original_names = {record["path"] for record in original_records}
    certificate = _record(root, "analysis/pairing_ablation/source_alignment_certificate.json")
    if certificate["sha256"] != CERTIFICATE_SHA256:
        raise AuditVerificationError("Historical alignment receipt changed")
    artifacts, sources = [], {}
    for label, directory, script in AUDITS:
        audit_manifest = json.loads(_path(root, f"{directory}/audit_manifest.json").read_text())
        input_records = audit_manifest.get("inputs", audit_manifest.get("input_inventory"))
        if not isinstance(input_records, list):
            raise AuditVerificationError("Missing original scalar source inventory")
        input_names = []
        for source in input_records:
            name = source["path"]
            if name not in original_names:
                if name != certificate["path"]:
                    raise AuditVerificationError("Nonpublic source in derived audit")
                continue
            record = _record(root, name)
            if any(record[key] != source[key] for key in ("bytes", "sha256")):
                raise AuditVerificationError("Source hash differs from audit provenance")
            sources[name] = record
            input_names.append(name)
        expected_names = {entry["path"] for entry in audit_manifest["outputs"]} | {
            "audit_manifest.json",
            "independent_verification.json",
            "main_vs_independent.csv",
        }
        actual_names = {path.name for path in (root / directory).iterdir()}
        if expected_names != actual_names:
            raise AuditVerificationError("Untracked or missing derived audit artifact")
        for name in sorted(expected_names):
            if PurePosixPath(name).name != name or Path(name).suffix not in {".csv", ".json"}:
                raise AuditVerificationError("Only scalar CSV/JSON audit artifacts are allowed")
            artifact = _record(root, f"{directory}/{name}")
            artifact.update(
                audit=label,
                derived=True,
                source_inputs=sorted(input_names),
                analysis_script=script,
                model_calls=0,
                role=(
                    "historical_provenance_receipt_readback_only"
                    if name == "source_alignment_certificate.json"
                    else "independent_verification"
                    if name in {"independent_verification.json", "main_vs_independent.csv"}
                    else "retrospective_scalar_derivation"
                ),
            )
            artifacts.append(artifact)
    return {
        "schema_version": 1,
        "scope": "retrospective public scalar audit",
        "source_public_commit": "af7b402839d1052ece994c80a91e59e10d7474dc",
        "derived": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "new_model_observations": 0,
        "predictor_refits": 0,
        "new_prospective_confirmation": 0,
        "original_evidence_manifest": original,
        "original_evidence_files_checked": len(original_records),
        "source_inputs": [sources[name] for name in sorted(sources)],
        "analysis_implementation": [_record(root, name) for name in IMPLEMENTATION],
        "artifacts": artifacts,
        "historical_alignment_receipt": certificate,
        "verification": "scripts/verify_public_audits.py",
        "limitations": [
            "Existing published scalars only; no raw model outputs are reproduced",
            "Historical alignment receipt is read back, not a new receiver/probe audit",
            "Stage-only rule and descriptive headroom subset are retrospective",
            "Secondary family is historical summary readback, not new decision recomputation",
            "Zero calls describes this scalar pipeline, not unrelated external processes",
        ],
    }


def verify_public_audits(root):
    """Read-only checks; explicit exceptions are preserved under Python -O."""
    root = Path(root).resolve()
    saved = json.loads(_path(root, MANIFEST_NAME).read_text())
    expected = manifest_from_files(root)
    if json.dumps(saved, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise AuditVerificationError("Public derived audit manifest mismatch")
    p0, _ = verify_baseline_audit(root, root / AUDITS[0][1])
    p1, _ = verify_pairing_audit(root, root / AUDITS[1][1])
    for label, directory, report in (("P0", AUDITS[0][1], p0), ("P1", AUDITS[1][1], p1)):
        if report["status"] != "PASS":
            raise AuditVerificationError(f"{label} independent scalar verification failed")
        previous = json.loads(_path(root, f"{directory}/independent_verification.json").read_text())
        if json.dumps(previous, sort_keys=True) != json.dumps(report, sort_keys=True):
            raise AuditVerificationError(f"{label} saved independent report differs")
    return {
        "passed": True,
        "scope": expected["scope"],
        "derived_artifacts": len(expected["artifacts"]),
        "original_evidence_files_checked": expected["original_evidence_files_checked"],
        "p0_independent_scalar_fields": p0["checked_scalar_fields"],
        "p1_independent_scalar_fields": p1["checked_scalar_fields"],
        "model_calls": 0,
        "gpu_calls": 0,
        "new_model_observations": 0,
        "predictor_refits": 0,
        "new_prospective_confirmation": 0,
    }
