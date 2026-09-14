"""Structural workflow guardrails, not evidence that hosted scanning succeeded."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_security_policy_keeps_private_reporting_and_research_boundary():
    text = (ROOT / "SECURITY.md").read_text()
    assert "current `main` branch" in text
    assert "latest public release, if any" in text
    assert "private vulnerability reporting" in text
    assert "Do not include real credentials, private keys" in text
    assert "do not create a software-security risk" in text
    assert "normal issues instead" in text
    assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)


def workflow(name):
    # BaseLoader avoids YAML 1.1 treating GitHub's `on` key as a boolean.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_workflows_pin_actions_and_avoid_privileged_pr_execution():
    for name in ("ci.yml", "codeql.yml"):
        config = workflow(name)
        assert "pull_request_target" not in config["on"]
        assert config["permissions"] == {"contents": "read"}
        for job in config["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", step["uses"])
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] == "false"


def test_cpu_checks_keep_matrix_and_explicit_audit_boundaries():
    job = workflow("ci.yml")["jobs"]["cpu"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert job["env"]["HF_HUB_OFFLINE"] == "1"
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    for command in (
        "python -m pytest -q",
        "python scripts/verify_evidence.py",
        "python -O scripts/verify_evidence.py",
        "python scripts/verify_public_audits.py",
        "python -O scripts/verify_public_audits.py",
        "python scripts/check_public_hygiene.py",
        "python examples/compact_dit.py",
        "python -m ruff check .",
        "python -m ruff format --check .",
        "PUBLIC_AUDIT_MANIFEST.json",
    ):
        assert command in commands


def test_codeql_is_static_python_with_minimal_write_scope():
    config = workflow("codeql.yml")
    assert config["on"]["schedule"]
    job = config["jobs"]["analyze"]
    assert job["permissions"] == {"contents": "read", "security-events": "write"}
    assert not any("run" in step for step in job["steps"])
    init = next(step for step in job["steps"] if "/init@" in step.get("uses", ""))
    assert init["with"]["languages"] == "python"
    assert init["with"]["build-mode"] == "none"


def test_dependabot_is_weekly_bounded_and_has_no_auto_merge():
    text = (ROOT / ".github/dependabot.yml").read_text()
    config = yaml.safe_load(text)
    assert config["version"] == 2
    assert {r["package-ecosystem"] for r in config["updates"]} == {"pip", "github-actions"}
    for row in config["updates"]:
        assert row["directory"] == "/"
        assert row["schedule"]["interval"] == "weekly"
        assert 1 <= row["open-pull-requests-limit"] <= 3
        assert "auto-merge" not in row
