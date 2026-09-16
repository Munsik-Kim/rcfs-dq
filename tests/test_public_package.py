"""Public packaging contracts; scanners are heuristic, not a security certification."""

import hashlib
import json
import re
import runpy
import shlex
import tomllib
from pathlib import Path

import rcfs_dq
from rcfs_dq.public_hygiene import verify_public_hygiene
from rcfs_dq.verification import verify_inventory

ROOT = Path(__file__).resolve().parents[1]


def test_protected_payload_hashes_and_complete_scope():
    manifest_path = ROOT / "PUBLIC_EVIDENCE_MANIFEST.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "8a5ffc01d6f673fcd46c4c9f404baadc4a76fa15e014c149740c00ce44e946ba"
    )
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == (
        "625ff72624cb754518038bf91a271898f93e0efe58e5082d2faaf260e229d422"
    )
    manifest = json.loads(manifest_path.read_text())
    assert verify_inventory(ROOT, manifest) == 22
    payload = {
        p.relative_to(ROOT).as_posix()
        for folder in ("configs", "evidence", "figures")
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
    }
    assert payload == {row["path"] for row in manifest["files"]}


def test_verifier_public_json_contract():
    result = runpy.run_path(str(ROOT / "scripts/verify_evidence.py"))["verify"]()
    assert set(result) == {
        "passed",
        "inventory_files",
        "scalar_comparisons",
        "maximum_scaled_discrepancy",
        "maximum_absolute_discrepancy",
        "confirmation_signs",
        "confirmation_stricter_replications",
        "maps",
        "primary_sets",
        "decision_units",
        "policies",
        "mean_regret",
        "scope",
        "model_calls",
    }
    assert result["passed"] and result["scalar_comparisons"] == 19192
    assert result["model_calls"] == 0
    assert (
        result["scope"] == "public scalar recomputation, not raw-terminal or full experiment replay"
    )


def test_no_operational_assignments_in_readme_or_roadmap():
    pattern = re.compile(r"authorized_to_(?:push|execute(?:_followup)?)\s*[=:]")
    for name in ("README.md", "docs/ROADMAP.md"):
        assert not pattern.search((ROOT / name).read_text()), name


def test_python_and_version_contract():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == rcfs_dq.__version__ == "0.1.1"
    assert project["requires-python"] == ">=3.11,<3.13"
    assert project["dependencies"] == ["numpy>=1.26,<2", "torch>=2.14,<2.15", "setuptools>=83,<84"]
    assert project["optional-dependencies"]["dit"] == ["diffusers==0.38.0"]


def test_readme_links_and_command_paths():
    readme = (ROOT / "README.md").read_text()
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        if "://" not in target and not target.startswith("#"):
            assert (ROOT / target.split("#")[0]).is_file(), target
    for line in readme.splitlines():
        if line.startswith("python "):
            tokens = shlex.split(line)
            if tokens[1].endswith(".py"):
                assert (ROOT / tokens[1]).is_file(), tokens[1]
            if "-c" in tokens and tokens[tokens.index("-c") + 1].endswith(".txt"):
                assert (ROOT / tokens[tokens.index("-c") + 1]).is_file()


def test_public_path_secret_and_large_file_hygiene():
    report = verify_public_hygiene(ROOT)
    assert report["passed"], report["findings"]  # Matched values are never included.
