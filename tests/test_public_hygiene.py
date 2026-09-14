"""Synthetic canaries are assembled at runtime; no credentials are stored here."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rcfs_dq.public_hygiene import MAX_FILE_BYTES, public_paths, scan_public_paths

ROOT = Path(__file__).resolve().parents[1]


def canaries():
    # Deliberate fragments avoid storing live-looking secrets or private paths.
    return [
        ("personal_path", "/" + "home" + "/fixture-person/data"),
        ("personal_path", "/" + "Users" + "/fixture-person/data"),
        ("personal_path", "/mnt/c/" + "Users" + "/fixture-person/data"),
        ("personal_path", "C:" + "\\Users" + "\\fixture-person\\data"),
        ("personal_path", "c:" + "/Users" + "/fixture-person/data"),
        ("private_cloud", "https://" + "drive.google" + ".com/file/d/fixture/view"),
        ("private_cloud", "https://" + "docs.google" + ".com/document/d/fixture/edit"),
        ("private_cloud", "/tmp/" + "codex-remote" + "-attachments/fixture"),
        ("token", "ghp_" + "X" * 36),
        ("token", "github_pat_" + "X" * 60),
        ("token", "sk-" + "X" * 48),
        ("token", "sk-" + "proj-" + "X" * 80),
        ("token", "AKIA" + "X" * 16),
        ("credential_assignment", 'api_key="' + "X" * 32 + '"'),
        ("private_key", "-----BEGIN " + "PRIVATE KEY-----"),
        ("private_key", "-----BEGIN " + "OPENSSH PRIVATE KEY-----"),
        ("private_key", "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----"),
    ]


@pytest.mark.parametrize("index", range(17))
@pytest.mark.parametrize("relative", ["analysis/p0p1/fixture.csv", "PUBLIC_AUDIT_MANIFEST.json"])
def test_sensitive_content_in_derived_audits_is_not_ignored(tmp_path, index, relative):
    category, content = canaries()[index]
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    findings = scan_public_paths(tmp_path, public_paths(tmp_path))
    assert {row["category"] for row in findings} == {category}
    assert all(row["path"] == relative for row in findings)
    assert content not in json.dumps(findings)


@pytest.mark.parametrize(
    "filename",
    ["model.pt", "model.PTH", "model.bin", "model.safetensors", "raw.npy", "raw.npz", "review.ZIP"],
)
def test_model_raw_archive_extensions_rejected(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(b"fixture")
    findings = scan_public_paths(tmp_path, [path])
    assert findings == [{"path": filename, "category": "excluded_payload"}]


@pytest.mark.parametrize("filename", [".env", ".env.local", ".env.example"])
def test_environment_files_rejected(tmp_path, filename):
    path = tmp_path / filename
    path.write_text("fixture")
    findings = scan_public_paths(tmp_path, [path])
    assert findings == [{"path": filename, "category": "environment_file"}]


def test_image_metadata_does_not_bypass_scanner(tmp_path):
    path = tmp_path / "figure.png"
    content = canaries()[8][1]
    path.write_bytes(b"\x89PNG\x00" + content.encode())
    findings = scan_public_paths(tmp_path, [path])
    assert findings == [{"path": "figure.png", "category": "token"}]
    assert content not in json.dumps(findings)


def test_sensitive_filename_is_redacted(tmp_path):
    content = canaries()[8][1]
    path = tmp_path / (content + ".txt")
    path.write_text("fixture")
    findings = scan_public_paths(tmp_path, [path])
    assert findings == [{"path": "[redacted path]", "category": "token"}]
    assert content not in json.dumps(findings)


def test_symlink_and_outside_root_are_rejected_without_reading(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    outside = tmp_path / "fixture.txt"
    outside.write_text("fixture")
    link = root / "link.txt"
    link.symlink_to(outside)
    findings = scan_public_paths(root, [link, outside])
    assert findings == [
        {"path": "link.txt", "category": "unsafe_path"},
        {"path": "[outside root]", "category": "unsafe_path"},
    ]


def test_missing_and_large_files_rejected(tmp_path):
    large = tmp_path / "large.csv"
    with large.open("wb") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    findings = scan_public_paths(tmp_path, [large, tmp_path / "missing.csv"])
    assert findings == [
        {"path": "large.csv", "category": "large_file"},
        {"path": "missing.csv", "category": "missing_or_nonregular_file"},
    ]


def test_nested_generated_name_is_not_an_audit_exclusion(tmp_path):
    path = tmp_path / "analysis/dist/data.csv"
    path.parent.mkdir(parents=True)
    path.write_text(canaries()[0][1])
    assert path in public_paths(tmp_path)
    assert scan_public_paths(tmp_path, public_paths(tmp_path))


def test_hygiene_cli_fails_without_printing_matched_value(tmp_path):
    category, content = canaries()[9]
    (tmp_path / "PUBLIC_AUDIT_MANIFEST.json").write_text(content)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_public_hygiene.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 1
    assert category in result.stdout
    assert content not in result.stdout + result.stderr
    assert not json.loads(result.stdout)["passed"]
