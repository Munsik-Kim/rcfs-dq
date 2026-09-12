"""Public packaging contracts; scanners are heuristic, not a security certification."""

import hashlib
import json
import re
import runpy
import shlex
import subprocess
import tomllib
from pathlib import Path

import rcfs_dq
from rcfs_dq.verification import verify_inventory

ROOT = Path(__file__).resolve().parents[1]


def public_paths():
    if (ROOT / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        return sorted({ROOT / p.decode() for p in output.split(b"\0") if p})
    # A clean source export has no Git metadata. Ignore local build/test products.
    ignored = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(
            part in ignored or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts
        )
    )


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
    assert project["dependencies"] == ["numpy>=1.26,<2", "torch>=2.11,<2.12"]
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
    # Pattern literals below are deliberate scanner definitions, not credentials.
    patterns = {
        "personal_path": re.compile(
            r"/(?:home|Users)/[\w.-]+/|/mnt/[a-z]/Users/|\b[A-Z]:[/\\]Users[/\\]"
        ),
        "private_cloud": re.compile(
            r"drive[.]google[.]com|docs[.]google[.]com|codex[-]remote[-]attachments"
        ),
        "token": re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{32,})\b"
        ),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    excluded_suffixes = {".pt", ".pth", ".safetensors", ".bin", ".npy", ".npz", ".zip"}
    problems = []
    for path in public_paths():
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or path.stat().st_size > 100_000_000:
            problems.append((rel, "symlink_or_large_file"))
        if path.suffix in excluded_suffixes:
            problems.append((rel, "excluded_payload"))
        if path.suffix == ".png":
            continue
        content = path.read_text(errors="replace")
        for category, pattern in patterns.items():
            if pattern.search(content):
                problems.append((rel, category))
    assert not problems, problems  # Never include a matched secret value in failures.
