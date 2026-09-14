import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rcfs_dq.decisions import Candidate, select
from rcfs_dq.pairing_audit import (
    CERTIFICATE_SHA256,
    aligned_geometry,
    export_audit,
    geometry_scores,
    sha256,
    stage_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]


def cells():
    return (Candidate("a", "g", 4, 4, 12), Candidate("b", "g", 15, 4, 12))


def test_stage_coefficient_equal_source_cells_not_group_means():
    candidates = (*cells(), Candidate("c", "h", 4, 8, 8))
    result = stage_coefficients(candidates, {"a": 4.0, "b": 1.0, "c": 16.0}, 0.5)
    assert result[4]["source_cell_count"] == 2
    assert result[4]["coefficient"] == pytest.approx(math.sqrt(8))
    assert result[15]["coefficient"] == 1


def test_stage_coefficient_sorted_input_order_invariance():
    assert stage_coefficients(cells(), {"a": 4.0, "b": 1.0}, 0.7) == stage_coefficients(
        reversed(cells()), {"b": 1.0, "a": 4.0}, 0.7
    )


@pytest.mark.parametrize(
    "g",
    [
        {"a": 1.0},
        {"a": 0.0, "b": 1.0},
        {"a": float("nan"), "b": 1.0},
        {"a": float("inf"), "b": 1.0},
    ],
)
def test_bad_geometry_rejected(g):
    with pytest.raises(ValueError):
        stage_coefficients(cells(), g, 0.7)


def test_duplicate_source_cell_rejected():
    with pytest.raises(ValueError):
        stage_coefficients((*cells(), cells()[0]), {"a": 4.0, "b": 1.0}, 0.7)


def test_equal_G_and_beta_zero_reduce_to_energy():
    energy = {"a": 2.0, "b": 3.0}
    for gains, beta in [({"a": 1.0, "b": 1.0}, 0.7), ({"a": 7.0, "b": 2.0}, 0.0)]:
        scores = geometry_scores(cells(), energy, gains, beta)
        assert scores == energy
        assert select(cells(), scores) == select(cells(), energy)


def test_same_G_argmin_does_not_imply_same_weighted_choice():
    energy = {"a": 1.0, "b": 3.0}
    g1, g2 = {"a": 2.0, "b": 1.0}, {"a": 10.0, "b": 1.0}
    assert select(cells(), g1) == select(cells(), g2) == "b"
    assert select(cells(), geometry_scores(cells(), energy, g1, 1.0)) == "a"
    assert select(cells(), geometry_scores(cells(), energy, g2, 1.0)) == "b"


def test_score_tie_respects_unchanged_source_order():
    assert select(tuple(reversed(cells())), {"a": 1.0, "b": 1.0}) == "a"


@pytest.mark.parametrize(
    "energy", [{"a": -1.0, "b": 1.0}, {"a": 1.0}, {"a": float("nan"), "b": 1.0}]
)
def test_invalid_energy(energy):
    with pytest.raises(ValueError):
        geometry_scores(cells(), energy, {"a": 4.0, "b": 1.0}, 0.7)


def test_no_outcome_argument_or_shared_mutable_coefficient_input():
    gains = {"a": 4.0, "b": 1.0}
    before = dict(gains)
    stage_coefficients(cells(), gains, 0.7)
    assert gains == before
    assert set(inspect.signature(stage_coefficients).parameters) == {
        "candidates",
        "geometry",
        "beta",
    }
    assert set(inspect.signature(geometry_scores).parameters) == {
        "members",
        "energies",
        "geometry",
        "beta",
    }


def test_historical_hash_fails_closed(tmp_path):
    directory = tmp_path / "analysis/pairing_ablation"
    directory.mkdir(parents=True)
    (directory / "source_alignment_certificate.json").write_text("{}\n")
    with pytest.raises(ValueError, match="certificate hash mismatch"):
        aligned_geometry(tmp_path, {}, {})


def test_fixed_aligned_source_certificate_and_scope():
    certificate = ROOT / "analysis/pairing_ablation/source_alignment_certificate.json"
    assert sha256(certificate) == CERTIFICATE_SHA256
    data = json.loads(certificate.read_text())
    assert data["status"] == "ALIGNED_HISTORICAL_TRIPLE"
    assert len(data["cells"]) == 81
    assert all(r["same_receiver_sets"] and r["same_alpha"] for r in data["cells"])
    assert sum(not r["frozen_actual_bitwise_equal"] for r in data["cells"]) == 10
    assert all(r["common_valid_pairs"] >= 22 for r in data["cells"])


def test_public_reproduction_stage_coefficients_without_main_function():
    import csv

    with (ROOT / "configs/candidates.csv").open() as stream:
        candidates = list(csv.DictReader(stream))
    frozen = json.loads((ROOT / "configs/frozen_predictor.json").read_text())
    with (ROOT / "analysis/pairing_ablation/stage_coefficients.csv").open() as stream:
        output = list(csv.DictReader(stream))
    for row in output:
        gains = [
            frozen["G"][c["cell_id"]] for c in candidates if int(c["stage"]) == int(row["stage"])
        ]
        reference = float(np.exp(frozen["beta"] * np.mean(np.log(gains))))
        assert len(gains) == int(row["source_cell_count"]) == 27
        assert float(row["coefficient"]) == pytest.approx(reference, abs=1e-12, rel=1e-10)


def test_export_refuses_protected_namespace_and_repository_ancestor():
    for destination in (ROOT, ROOT.parent, ROOT / "evidence"):
        with pytest.raises(ValueError):
            export_audit(ROOT, destination)


def test_export_refuses_nonempty_external_and_symlink(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "user.txt").write_text("user content")
    with pytest.raises(ValueError, match="new or empty"):
        export_audit(ROOT, occupied)
    link = tmp_path / "link"
    link.symlink_to(occupied, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        export_audit(ROOT, link)
    assert (occupied / "user.txt").read_text() == "user content"


def test_temp_export_and_readonly_reproduction(tmp_path):
    destination = tmp_path / "audit"
    result = export_audit(ROOT, destination)
    assert result["primary_units"] == 1728
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in destination.iterdir()}
    assert export_audit(ROOT, destination, check=True)["status"] == "PASS_READONLY"
    assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in destination.iterdir()}
    (destination / "decisions.csv").write_text("corrupt\n")
    with pytest.raises(AssertionError, match="Recomputed P1 output mismatch"):
        export_audit(ROOT, destination, check=True)
