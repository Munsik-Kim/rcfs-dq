"""Public claims must match derived scalar records, including unfavorable results."""

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "analysis/baseline_audit"
P1 = ROOT / "analysis/pairing_ablation"
POLICIES = {
    "Energy": "energy",
    "Source Constant": "constant",
    "Latest-stage": "latest_stage",
    "Pure-G": "pure_G",
    "Candidate-B": "candidate_B",
    "Hybrid": "hybrid",
    "Uniform-random expectation": "uniform_random_expected",
}
GEOMETRY = {
    "Actual": "aligned_actual_source",
    "Donor": "aligned_donor_source",
    "Isotropic": "aligned_isotropic_source",
    "Stage-only": "stage_only",
}
P0_CONTRASTS = {
    "Candidate-B − Energy": ("candidate_B", "energy"),
    "Candidate-B − Constant": ("candidate_B", "constant"),
    "Candidate-B − Latest-stage": ("candidate_B", "latest_stage"),
}
P1_CONTRASTS = {
    "Actual − Stage-only": ("aligned_actual_source", "stage_only"),
    "Actual − Donor": ("aligned_actual_source", "aligned_donor_source"),
    "Donor − Isotropic": ("aligned_donor_source", "aligned_isotropic_source"),
    "Actual − Isotropic": ("aligned_actual_source", "aligned_isotropic_source"),
}
DOCUMENTS = (
    "README.md",
    "docs/RESULTS.md",
    "docs/BASELINE_AUDIT.md",
    "docs/PAIRING_ABLATION.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ROADMAP.md",
    "CHANGELOG.md",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def markdown_rows(document, first_column):
    lines = text(document).splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(f"| {first_column} |")]
    assert len(starts) == 1
    rows = {}
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        key, *fields = [cell.strip() for cell in line.strip("|").split("|")]
        assert key not in rows
        rows[key] = fields
    return rows


def all_table_rows(document):
    rows = {}
    for line in text(document).splitlines():
        if line.startswith("|"):
            key, *values = [cell.strip() for cell in line.strip("|").split("|")]
            rows.setdefault(key, []).append(values)
    return rows


def signed(value):
    return f"{float(value):+.6f}".replace("-", "−")


def interval(row):
    return f"{signed(row['point'])} [{signed(row['low'])}, {signed(row['high'])}]"


@pytest.mark.parametrize("document", ["README.md", "docs/RESULTS.md"])
def test_policy_table_is_rounded_from_recomputed_records(document):
    rows = markdown_rows(document, "Policy") if document == "README.md" else None
    if rows is None:
        # RESULTS contains a separate descriptive tail table under the same label.
        section = text(document).split("### P0:", 1)[1].split("Latest-stage chooses", 1)[0]
        rows = {}
        for line in section.splitlines():
            fields = [cell.strip() for cell in line.strip("|").split("|")]
            if fields[0] in POLICIES:
                rows[fields[0]] = fields[1:]
    expected = (
        {"Energy", "Source Constant", "Latest-stage", "Candidate-B"}
        if document == "README.md"
        else set(POLICIES)
    )
    assert set(rows) == expected
    source = {(r["policy"], r["metric"]): r for r in read_csv(P0 / "baseline_results.csv")}
    for label, values in rows.items():
        assert values == [
            f"{float(source[POLICIES[label], metric]['point']):.6f}"
            for metric in ("absolute_regret", "normalized_regret")
        ]
    if document == "README.md":
        assert sum(line.startswith("|---") for line in text(document).splitlines()) == 1
        assert "95%" not in text(document)  # Keep detailed bootstrap tables in RESULTS.


def test_p0_paired_absolute_and_normalized_claims_match_records():
    documented = all_table_rows("docs/RESULTS.md")
    source = {
        (r["left"], r["right"], r["metric"]): r for r in read_csv(P0 / "paired_comparisons.csv")
    }
    for label, pair in P0_CONTRASTS.items():
        assert documented[label] == [
            [interval(source[*pair, metric]) for metric in ("absolute_regret", "normalized_regret")]
        ]


@pytest.mark.parametrize("document", ["docs/RESULTS.md", "docs/PAIRING_ABLATION.md"])
def test_all_four_p1_contrasts_and_score_means_match_records(document):
    documented = all_table_rows(document)
    source = {
        (r["left"], r["right"], r["metric"]): r for r in read_csv(P1 / "paired_comparisons.csv")
    }
    for label, pair in P1_CONTRASTS.items():
        assert documented[label] == [
            [interval(source[*pair, metric]) for metric in ("absolute_regret", "normalized_regret")]
        ]
    scores = {(r["policy"], r["metric"]): r for r in read_csv(P1 / "score_results.csv")}
    table = markdown_rows(document, "Geometry score")
    assert set(table) == set(GEOMETRY)
    for label, policy in GEOMETRY.items():
        assert table[label] == [
            f"{float(scores[policy, metric]['point']):.6f}"
            for metric in ("absolute_regret", "normalized_regret")
        ]


def test_descriptive_tails_and_oracle_hits_match_records():
    documented = all_table_rows("docs/RESULTS.md")
    tails = {
        r["policy"]: r
        for r in read_csv(P0 / "policy_tails.csv")
        if r["set_id"] == "ALL" and r["metric"] == "absolute_regret"
    }
    hit = {
        r["policy"]: r for r in read_csv(P0 / "baseline_results.csv") if r["metric"] == "oracle_hit"
    }
    for label in ("Energy", "Source Constant", "Latest-stage", "Candidate-B"):
        policy = POLICIES[label]
        expected = [f"{float(hit[policy]['point']):.6f}"] + [
            f"{float(tails[policy][field]):.6f}" for field in ("Q90", "Q95", "Q99", "max")
        ]
        assert expected in documented[label]


def test_headroom_claims_retain_full_grid_and_exploratory_boundary():
    rows = read_csv(P0 / "oracle_headroom.csv")
    assert len(rows) == 27
    assert sum(r["majority_stage"] == "15" for r in rows) == 25
    assert sum(r["fully_fixed"] == "True" for r in rows) == 14
    assert sum(float(r["majority_rate"]) < 0.75 for r in rows) == 5
    assert sum(float(r["majority_rate"]) >= 0.75 for r in rows) == 22
    results = text("docs/RESULTS.md")
    assert "25/27" in results and "14/27" in results
    assert "320 units" in results and "exploratory" in results
    assert "not a new primary PASS criterion" in results
    assert "22/27" in text("docs/BASELINE_AUDIT.md")


@pytest.mark.parametrize("document", DOCUMENTS)
def test_public_document_links_and_script_paths_exist(document):
    path = ROOT / document
    content = text(document)
    for target in re.findall(r"\]\(([^)]+)\)", content):
        if "://" in target or target.startswith("#"):
            continue
        assert (path.parent / target.split("#", 1)[0]).is_file(), (document, target)
    for relative in re.findall(r"\bpython\s+((?:scripts|examples)/[\w./-]+\.py)", content):
        assert (ROOT / relative).is_file(), (document, relative)


@pytest.mark.parametrize(
    "document",
    ["README.md", "docs/RESULTS.md", "docs/BASELINE_AUDIT.md", "docs/PAIRING_ABLATION.md"],
)
def test_retrospective_zero_execution_scope_is_explicit(document):
    content = " ".join(text(document).lower().split())
    assert "retrospective" in content and "cpu-only" in content
    for field in (
        "model calls",
        "gpu calls",
        "new model observations",
        "predictor refits",
        "new prospective confirmation",
    ):
        assert f"{field} = 0" in content


def test_negative_result_and_absolute_risk_limit_are_visible():
    readme = text("README.md")
    results = text("docs/RESULTS.md")
    assert "E1-Tail0" in readme and "failed" in readme
    assert "3/474/76/1175" in results
    assert "FP32 is not ground truth" in readme
    assert "additional adaptive utility" in results.lower()
    assert "not established" in readme
    assert "lower\n  mean absolute regret" in readme


def test_roadmap_priority_and_endpoint_are_not_old_stage_selection():
    roadmap = text("docs/ROADMAP.md")
    assert roadmap.index("## 1. P2:") < roadmap.index("## 2. Real packed")
    assert roadmap.index("## 2. Real packed") < roadmap.index("## 3. Allocation")
    normalized = " ".join(roadmap.split())
    assert "Hold stage, exact cost and precision fixed while candidate group varies" in normalized
    assert "primary endpoint is **mean absolute regret**" in normalized
    assert "future protocol" in normalized and "not an executed experiment" in normalized
    assert "Q95, Q99" in normalized
