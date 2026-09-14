"""Independent NumPy/CSV reference for the retrospective free-baseline audit.

This module does not import the main audit, scoring, selection, or aggregation
implementation.  Its inputs are public scalar observations and frozen registries,
not original terminal tensors.  Passing does not reproduce model execution,
leakage checks, or the validity of historical geometry measurements.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

POLICIES = (
    "constant",
    "energy",
    "candidate_B",
    "hybrid",
    "latest_stage",
    "stage_position_only",
    "pure_G",
    "uniform_random_expected",
)
METRICS = ("absolute_regret", "normalized_regret", "oracle_hit", "all_minima_hit")
ATOL, RTOL = 1e-12, 1e-10
BOOTSTRAP_SEED = 2026091101
BOOTSTRAP_REPLICATES = 10000


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicate CSV columns")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    return rows


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inventory(root, records):
    """Readback of a finite relative-path inventory, independently implemented."""
    root = Path(root).resolve()
    paths = set()
    for record in records:
        name = record["path"]
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or name in paths
        ):
            raise ValueError("Unsafe or duplicate inventory path")
        paths.add(name)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Inventory symlink or escape")
        if path.stat().st_size != int(record["bytes"]) or _hash(path) != record["sha256"]:
            raise ValueError(f"Source inventory mismatch: {name}")
    return len(paths)


def _finite(value, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Expected finite nonnegative scalar (positive when specified)")
    return value


def independent_choices(members, energies, geometry, beta, constant):
    """Choose from features only; the API intentionally has no terminal-risk argument."""
    ids = [r["cell_id"] for r in members]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(energies):
        raise ValueError("Exact unique candidate/energy coverage required")
    if not set(ids) <= set(geometry) or constant not in ids or not math.isfinite(beta):
        raise ValueError("Missing frozen coefficients/constant or invalid beta")
    energy = {name: _finite(energies[name]) for name in ids}
    gain = {name: _finite(geometry[name], positive=True) for name in ids}
    scores = {name: _finite(energy[name] * gain[name] ** beta) for name in ids}
    ordered = sorted(
        members, key=lambda r: (r["stage"], r["group_order"], -r["bits"], r["cell_id"])
    )
    order = [r["cell_id"] for r in ordered]
    # Stable min over the independently reconstructed canonical order resolves ties.
    chosen = {
        "constant": constant,
        "energy": min(order, key=energy.__getitem__),
        "candidate_B": min(order, key=scores.__getitem__),
        "pure_G": min(order, key=gain.__getitem__),
        "latest_stage": next(
            r["cell_id"] for r in ordered if r["stage"] == max(v["stage"] for v in ordered)
        ),
        "stage_position_only": min(ordered, key=lambda r: -r["stage"])["cell_id"],
        "hybrid": min(
            ordered,
            key=lambda r: (
                r["regime"] == "M3",
                energy[r["cell_id"]] if r["regime"] == "M3" else scores[r["cell_id"]],
            ),
        )["cell_id"],
        "uniform_random_expected": None,
    }
    return chosen


def independent_evaluate(order, values, selected):
    """FP64 expected regret, retaining unresolved normalized values as None."""
    if len(order) < 2 or len(set(order)) != len(order) or set(order) != set(values):
        raise ValueError("Exact unique risk coverage required")
    y = np.asarray([_finite(values[name]) for name in order], dtype=np.float64)
    if selected is not None and selected not in order:
        raise ValueError("Unknown selected candidate")
    j = int(np.argmin(y))
    low, high = float(y[j]), float(np.max(y))
    span, tau = high - low, 1e-12 * max(high, 1.0)
    if selected is None:
        selected_y = float(np.mean(y))
        hit = 1.0 / len(order)
        minima_hit = float(np.count_nonzero(y == low)) / len(order)
        probabilities = {name: 1.0 / len(order) for name in order}
    else:
        selected_y = float(y[order.index(selected)])
        hit = float(selected == order[j])
        minima_hit = float(selected_y == low)
        # Sparse deterministic distribution; omitted members have probability zero.
        probabilities = {selected: 1.0}
    regret = selected_y - low
    return dict(
        selected=selected,
        oracle=order[j],
        selected_Y=selected_y,
        oracle_Y=low,
        absolute_regret=regret,
        normalized_regret=regret / span if span > tau else None,
        oracle_hit=hit,
        all_minima_hit=minima_hit,
        risk_range=span,
        range_tau=tau,
        resolved=span > tau,
        selected_probability_json=probabilities,
    )


def independent_class_means(decisions, classes, seeds, set_ids, policy, metric):
    """Explicit set -> seed -> class aggregation; never silently complete-case."""
    lookup = {}
    for row in decisions:
        if row["policy"] != policy or row["set_id"] not in set_ids:
            continue
        key = row["class_id"], row["seed"], row["set_id"]
        if key in lookup:
            raise ValueError("Duplicate decision unit")
        lookup[key] = row[metric]
    expected = {(c, s, b) for c in classes for s in seeds for b in set_ids}
    if set(lookup) != expected:
        raise ValueError("Decision unit coverage differs from the fixed grid")
    if any(value is None for value in lookup.values()):
        return None
    return np.asarray(
        [np.mean([np.mean([lookup[c, s, b] for b in set_ids]) for s in seeds]) for c in classes],
        dtype=np.float64,
    )


def independent_interval(values, indices):
    values, indices = np.asarray(values, dtype=np.float64), np.asarray(indices)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty class vector required")
    if (
        indices.ndim != 2
        or indices.shape[1] != len(values)
        or not indices.size
        or indices.dtype.kind not in "iu"
        or indices.min() < 0
        or indices.max() >= len(values)
    ):
        raise ValueError("Invalid paired class indices")
    samples = np.mean(values[indices], axis=1)
    quantiles = np.quantile(samples, [0.025, 0.975, 0.05, 0.95], method="linear")
    return dict(
        zip(
            ("point", "low", "high", "lower95", "upper95"),
            [float(np.mean(values)), *map(float, quantiles)],
            strict=True,
        )
    )


def independent_headroom(order, risk_matrix):
    """Test-best-fixed diagnostics, with independently optimized absolute/NR objectives."""
    y = np.asarray(risk_matrix, dtype=np.float64)
    if (
        y.ndim != 2
        or not len(y)
        or y.shape[1] != len(order)
        or len(set(order)) != len(order)
        or not np.isfinite(y).all()
        or (y < 0).any()
    ):
        raise ValueError("Invalid rectangular risk matrix")
    oracle = np.argmin(y, axis=1)
    counts = np.bincount(oracle, minlength=len(order))
    frequency = counts[counts > 0] / len(y)
    minima = np.min(y, axis=1)
    regret = y - minima[:, None]
    # Full class/seed grid has equal weights. Modal oracle and risk optimum differ.
    abs_choice = int(np.argmin(np.mean(y, axis=0)))
    span = np.max(y, axis=1) - minima
    tau = 1e-12 * np.maximum(np.max(y, axis=1), 1)
    nr_choice, nr_value = None, None
    if np.all(span > tau):
        nr_means = np.mean(regret / span[:, None], axis=0)
        nr_index = int(np.argmin(nr_means))
        nr_choice, nr_value = order[nr_index], float(nr_means[nr_index])
    return dict(
        majority_candidate=order[int(np.argmax(counts))],
        majority_rate=float(np.max(counts) / len(y)),
        oracle_entropy_bits=float(-np.sum(frequency * np.log2(frequency))),
        fully_fixed=bool(np.count_nonzero(counts) == 1),
        oracle_tie_units=int(np.count_nonzero(np.sum(y == minima[:, None], axis=1) > 1)),
        empirical_best_fixed_abs_candidate=order[abs_choice],
        H_abs=float(np.mean(y[:, abs_choice]) - np.mean(minima)),
        empirical_best_fixed_nr_candidate=nr_choice,
        H_normalized=nr_value,
    )


def reference_from_sources(root, *, enforce_historical_grid=True):
    """Read original scalar inputs and calculate all reference decisions independently."""
    root = Path(root)
    inventory = json.loads((root / "PUBLIC_EVIDENCE_MANIFEST.json").read_text())
    validated = validate_inventory(root, inventory["files"])
    registry = _rows(root / "configs/candidates.csv")
    candidates, group_order = {}, {}
    for row in registry:
        name = row["cell_id"]
        if name in candidates:
            raise ValueError("Duplicate candidate registry identity")
        group_order.setdefault(row["group"], len(group_order))
        record = {
            **row,
            **{k: int(row[k]) for k in ("stage", "bits", "numel", "packed_bits", "saving_bits")},
            "group_order": group_order[row["group"]],
        }
        n, bits = record["numel"], record["bits"]
        if (
            n <= 0
            or not 2 <= bits <= 16
            or record["stage"] < 0
            or record["regime"] not in {"M3", "L78"}
            or record["packed_bits"] != n * bits
            or record["saving_bits"] != n * (16 - bits)
        ):
            raise ValueError("Candidate cost, stage, or regime mismatch")
        shapes = json.loads(row["shapes"])
        if not shapes or any(
            not s or any(type(d) is not int or d <= 0 for d in s) for s in shapes.values()
        ):
            raise ValueError("Invalid source weight shapes")
        if sum(math.prod(s) for s in shapes.values()) != n:
            raise ValueError("Weight-shape numel mismatch")
        candidates[name] = record
    frozen = json.loads((root / "configs/frozen_predictor.json").read_text())
    if set(frozen["G"]) != set(candidates) or frozen.get("fit_allowed") is not False:
        raise ValueError("Frozen geometry coverage or fit authorization mismatch")
    sets, set_counts = defaultdict(list), {}
    for row in _rows(root / "configs/cost_sets.csv"):
        key = row["family"], row["set_id"]
        if key in set_counts and set_counts[key] != int(row["candidate_count"]):
            raise ValueError("Inconsistent declared set size")
        set_counts[key] = int(row["candidate_count"])
        c = candidates[row["cell_id"]]
        if c["cell_id"] in {v["cell_id"] for v in sets[key]}:
            raise ValueError("Duplicate cost-set member")
        if c["saving_bits"] != int(row["saving_bits"]) or c["packed_bits"] != int(
            row["packed_bits"]
        ):
            raise ValueError("Cost registry mismatch")
        sets[key].append(c)
    for (family, set_id), members in sets.items():
        if len(members) != set_counts[family, set_id]:
            raise ValueError("Declared cost-set coverage mismatch")
        if len(members) < 2 or len({r["saving_bits"] for r in members}) != 1:
            raise ValueError("Cost set is not exact matched-saving")
        if family == "primary" and len({(r["group"], r["bits"]) for r in members}) != 1:
            raise ValueError("Primary set mixes group or precision")
        members.sort(key=lambda r: (r["stage"], r["group_order"], -r["bits"], r["cell_id"]))
    derived = defaultdict(set)
    for name, c in candidates.items():
        derived[c["group"], c["bits"], c["saving_bits"]].add(name)
    expected_sets = {frozenset(v) for v in derived.values() if len(v) >= 2}
    primary_sets = {
        frozenset(r["cell_id"] for r in v) for (family, _), v in sets.items() if family == "primary"
    }
    if expected_sets != primary_sets:
        raise ValueError("Incomplete or replaced exact primary membership")
    observations = {}
    score_residuals = []
    for row in _rows(root / "evidence/e1pred2r2/candidate_observations.csv"):
        key = int(row["class_id"]), int(row["seed"]), row["cell_id"]
        if key in observations or key[2] not in candidates:
            raise ValueError("Duplicate or unknown observation candidate")
        energy, y = _finite(row["energy"]), _finite(row["native_Y"])
        score = energy * _finite(frozen["G"][key[2]], positive=True) ** float(frozen["beta"])
        given = _finite(row["source_B_score"])
        if abs(score - given) > ATOL + RTOL * abs(given):
            raise ValueError("Frozen source Candidate-B score mismatch")
        score_residuals.append(abs(score - given))
        observations[key] = {"energy": energy, "Y": y}
    classes = sorted({k[0] for k in observations})
    seeds = sorted({k[1] for k in observations})
    grid = {(c, s, name) for c in classes for s in seeds for name in candidates}
    if set(observations) != grid:
        raise ValueError("Incomplete Cartesian observation grid")
    if enforce_historical_grid and (
        len(candidates),
        len(classes),
        len(seeds),
        len(primary_sets),
        len(observations),
    ) != (81, 16, 4, 27, 5184):
        raise ValueError("Historical evaluation scope differs from locked grid")
    decisions, headroom = [], []
    for (family, set_id), members in sorted(sets.items()):
        order = [r["cell_id"] for r in members]
        matrix = []
        for class_id in classes:
            for seed in seeds:
                energy = {name: observations[class_id, seed, name]["energy"] for name in order}
                y = {name: observations[class_id, seed, name]["Y"] for name in order}
                choices = independent_choices(
                    members, energy, frozen["G"], frozen["beta"], frozen["constants"][set_id]
                )
                matrix.append([y[name] for name in order])
                for policy, choice in choices.items():
                    result = independent_evaluate(order, y, choice)
                    decisions.append(
                        {
                            "family": family,
                            "class_id": class_id,
                            "seed": seed,
                            "set_id": set_id,
                            "policy": policy,
                            **result,
                            "candidate_count": len(order),
                            "saving_bits": members[0]["saving_bits"],
                            "selected_stage": candidates[choice]["stage"] if choice else None,
                            "oracle_stage": candidates[result["oracle"]]["stage"],
                        }
                    )
        h = independent_headroom(order, matrix)
        headroom.append(
            {
                "family": family,
                "set_id": set_id,
                **h,
                "majority_stage": candidates[h["majority_candidate"]]["stage"],
            }
        )
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(classes), (BOOTSTRAP_REPLICATES, len(classes))
    )
    class_means, summaries = {}, []
    for family in sorted({key[0] for key in sets}):
        set_ids = sorted(key[1] for key in sets if key[0] == family)
        for policy in POLICIES:
            for metric in METRICS:
                values = independent_class_means(decisions, classes, seeds, set_ids, policy, metric)
                class_means[family, policy, metric] = values
                interval = (
                    independent_interval(values, indices)
                    if values is not None
                    else {field: None for field in ("point", "low", "high", "lower95", "upper95")}
                )
                summaries.append({"family": family, "policy": policy, "metric": metric, **interval})
    return dict(
        decisions=decisions,
        headroom=headroom,
        summaries=summaries,
        class_means=class_means,
        indices=indices,
        classes=classes,
        seeds=seeds,
        sets=sets,
        candidates=candidates,
        inventory_files=validated,
        source_score_max_abs_error=max(score_residuals, default=0),
        frozen_beta=float(frozen["beta"]),
        source_inventory=inventory,
        bootstrap_indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
    )


class Comparison:
    """All rows are checked; compact output retains each column's worst keyed discrepancy."""

    def __init__(self):
        self.counts = Counter()
        self.maximum = {}
        self.failures = []

    def check(self, table, key, field, actual, expected):
        name = table, field
        self.counts[name] += 1
        numeric = isinstance(expected, (float, np.floating))
        if numeric:
            try:
                observed = float(actual)
            except (TypeError, ValueError):
                observed = math.nan
            difference = abs(observed - float(expected))
            tolerance = ATOL + RTOL * abs(float(expected))
            ok = math.isfinite(observed) and difference <= tolerance
            scaled = difference / tolerance if math.isfinite(difference) else math.inf
        else:
            if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
                try:
                    observed = int(actual)
                    if str(observed) != str(actual):
                        observed = actual
                except (TypeError, ValueError):
                    observed = actual
            elif expected is None:
                observed = None if actual in (None, "", "null") else actual
            elif isinstance(expected, bool):
                observed = {"True": True, "False": False, "true": True, "false": False}.get(
                    str(actual), actual
                )
            elif isinstance(expected, dict):
                try:
                    observed = json.loads(actual) if isinstance(actual, str) else actual
                except (ValueError, TypeError):
                    observed = actual
            else:
                observed = actual
            ok = observed == expected
            if isinstance(expected, bool):
                ok = type(observed) is bool and ok
            elif isinstance(expected, (int, np.integer)):
                ok = type(observed) is int and ok
            difference, tolerance, scaled = 0 if ok else 1, 0, 0 if ok else math.inf
        record = dict(
            table=table,
            metric=field,
            key=json.dumps(key, ensure_ascii=True),
            main=observed,
            independent=expected,
            abs_diff=difference,
            relative_diff=(
                difference / max(abs(float(expected)), abs(observed), 1e-12)
                if numeric and math.isfinite(observed)
                else None
            ),
            tolerance=tolerance,
            scaled_error=scaled,
            passed=bool(ok),
        )
        if name not in self.maximum or scaled > self.maximum[name]["scaled_error"]:
            self.maximum[name] = record
        if not ok and len(self.failures) < 100:
            self.failures.append(record)

    def table(self, name, observed, expected, key_fields, fields=None):
        actual_map = {}
        for row in observed:
            key = tuple(str(row[k]) for k in key_fields)
            if key in actual_map:
                raise ValueError(f"Duplicate main row in {name}: {key}")
            actual_map[key] = row
        expected_map = {tuple(str(row[k]) for k in key_fields): row for row in expected}
        if len(expected_map) != len(expected) or set(actual_map) != set(expected_map):
            raise ValueError(f"Main/reference row scope mismatch: {name}")
        for key, row in expected_map.items():
            for field in fields or [f for f in row if f not in key_fields]:
                if field not in actual_map[key]:
                    raise ValueError(f"Missing main field: {name}/{field}")
                self.check(name, key, field, actual_map[key][field], row[field])


def validate_baseline_contract(manifest, reference):
    """Compare declared scope to independently reconstructed support and fixed rules.

    Zero calls/fits are the contract of this scalar-only program, not a general
    independent certification that arbitrary external processes made no calls.
    """
    classes, seeds = reference["classes"], reference["seeds"]
    primary = sorted(s for family, s in reference["sets"] if family == "primary")
    eligible = sorted(
        r["set_id"]
        for r in reference["headroom"]
        if r["family"] == "primary" and r["majority_rate"] < 0.75
    )
    eligible_members = [r for s in eligible for r in reference["sets"]["primary", s]]
    expected = {
        "schema_version": 1,
        "status": "COMPLETED_PUBLIC_SCALAR_AUDIT",
        "scope": "retrospective_public_scalar_reanalysis",
        "classes": classes,
        "seeds": seeds,
        "candidates": len(reference["candidates"]),
        "primary_sets": len(primary),
        "decision_units_per_policy": len(classes) * len(seeds) * len(primary),
        "policy_rows": len(reference["decisions"]),
        "policies": list(POLICIES),
        "beta": reference["frozen_beta"],
        "scientific_model_calls": 0,
        "gpu_calls": 0,
        "fitted_parameters": 0,
        "unchanged_historical_decisions": True,
        "source_base_commit": reference["source_inventory"]["source_base_commit"],
        "protected_inventory_files_checked": reference["inventory_files"],
        "source_score_checks": len(classes) * len(seeds) * len(reference["candidates"]),
        "maximum_source_score_abs_error": reference["source_score_max_abs_error"],
        "unresolved_decision_rows": sum(not r["resolved"] for r in reference["decisions"]),
        "secondary_status": "source_summary_only",
        "aggregation": "equal cost sets per class/seed; equal four seeds; equal 16 classes",
        "headroom_aggregation": (
            "equal class/seed within each cost set; test-best fixed diagnostic only"
        ),
        "normalized_range_rule": "span > 1e-12*max(max(native_Y),1); undefined otherwise",
        "tie_rule": (
            "lower denoising stage index, source first-seen group order, higher bits, cell ID"
        ),
        "bootstrap": {
            "byte_contract": "row-major little-endian int64 without header",
            "draws": BOOTSTRAP_REPLICATES,
            "dtype": "int64",
            "generator": "PCG64",
            "index_matrix_sha256": reference["bootstrap_indices_sha256"],
            "one_sided": [0.05, 0.95],
            "quantile_method": "linear",
            "seed": BOOTSTRAP_SEED,
            "shape": [BOOTSTRAP_REPLICATES, len(classes)],
            "shared_paired_indices": True,
            "two_sided": [0.025, 0.975],
        },
        "exploratory_subset": {
            "bits": sorted({r["bits"] for r in eligible_members}),
            "groups": sorted({r["group"] for r in eligible_members}),
            "prospective": False,
            "selection": "test oracle majority < 0.75",
            "set_count": len(eligible),
            "sets": eligible,
            "units": len(eligible) * len(classes) * len(seeds),
        },
    }
    for field, value in expected.items():
        if field not in manifest or json.dumps(manifest[field], sort_keys=True) != json.dumps(
            value, sort_keys=True
        ):
            raise ValueError(f"Manifest semantic mismatch: {field}")
    return len(expected)


def validate_source_metadata(root, records, original_inventory):
    """Exact source linkage and typed schema checks, in addition to file SHA/bytes."""
    root = Path(root)
    originals = {r["path"]: r for r in original_inventory["files"]}
    commits = set()
    for row in records:
        path = root / row["path"]
        original = originals.get(row["path"])
        if original is None:
            raise ValueError("New P0 source is outside the original public inventory")
        for field in (
            "sha256",
            "bytes",
            "source_commit",
            "source_file_sha256",
            "column_projection",
        ):
            if field in original and row.get(field) != original[field]:
                raise ValueError(f"Source provenance metadata mismatch: {row['path']}/{field}")
        payload = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if row.get("public_blob") != blob:
            raise ValueError("Public Git blob does not match source bytes")
        commit = row.get("public_commit", "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Invalid source commit identifier")
        commits.add(commit)
        if path.suffix == ".csv":
            data = _rows(path)
            if row.get("rows") != len(data) or row.get("schema") != list(data[0]):
                raise ValueError("Source CSV schema or row count differs from manifest")
    if len(commits) != 1:
        raise ValueError("Source public commits are inconsistent")
    return next(iter(commits))


def verify_baseline_audit(root, audit_directory):
    """Fail-closed comparison with independently calculated scalar references."""
    root, audit = Path(root), Path(audit_directory)
    reference = reference_from_sources(root)
    comparison = Comparison()
    manifest = json.loads((audit / "audit_manifest.json").read_text())
    manifest_checks = validate_baseline_contract(manifest, reference)
    input_count = validate_inventory(root, manifest["inputs"])
    public_source = validate_source_metadata(
        root, manifest["inputs"], reference["source_inventory"]
    )
    if manifest["public_source_commit"] != public_source:
        raise ValueError("Manifest source commit differs from input provenance")
    output_count = validate_inventory(audit, manifest["outputs"])
    validate_inventory(root, manifest["implementation"])
    for record in manifest["outputs"]:
        if record["rows"] != len(_rows(audit / record["path"])):
            raise ValueError("Output inventory row count differs from saved table")
    if manifest["original_public_manifest_sha256"] != _hash(root / "PUBLIC_EVIDENCE_MANIFEST.json"):
        raise ValueError("Public source inventory identity changed")
    required_sources = {
        "configs/candidates.csv",
        "configs/cost_sets.csv",
        "configs/frozen_predictor.json",
        "evidence/e1pred2r2/candidate_observations.csv",
        "evidence/e1pred2r2/matched_cost_regret_summary.csv",
        "evidence/e1pred2r2/predictor_pairwise_comparison.csv",
        "evidence/e1pred2r2/decision_diversity.csv",
    }
    if not required_sources <= {r["path"] for r in manifest["inputs"]}:
        raise ValueError("Required source lock input missing")
    comparison.check(
        "manifest",
        "bootstrap",
        "indices_sha256",
        manifest["bootstrap"]["index_matrix_sha256"],
        reference["bootstrap_indices_sha256"],
    )
    comparison.check("manifest", "bootstrap", "seed", manifest["bootstrap"]["seed"], BOOTSTRAP_SEED)
    comparison.check(
        "manifest", "bootstrap", "draws", manifest["bootstrap"]["draws"], BOOTSTRAP_REPLICATES
    )
    comparison.table(
        "decisions",
        _rows(audit / "decisions.csv"),
        reference["decisions"],
        ("family", "class_id", "seed", "set_id", "policy"),
    )
    summaries = _rows(audit / "baseline_results.csv")
    if not {"absolute_regret", "normalized_regret", "oracle_hit"} <= {
        r["metric"] for r in summaries
    }:
        raise ValueError("Required baseline estimand missing")
    expected_summaries = [
        r for r in reference["summaries"] if r["metric"] in {row["metric"] for row in summaries}
    ]
    comparison.table(
        "baseline_results", summaries, expected_summaries, ("family", "policy", "metric")
    )
    comparison.table(
        "oracle_headroom",
        _rows(audit / "oracle_headroom.csv"),
        reference["headroom"],
        ("family", "set_id"),
    )
    pairs = _rows(audit / "paired_comparisons.csv")
    expected_pairs = []
    for row in pairs:
        key = row["family"], row["metric"]
        left = reference["class_means"][key[0], row["left"], key[1]]
        right = reference["class_means"][key[0], row["right"], key[1]]
        if left is None or right is None:
            raise ValueError("Unresolved primary class contrast")
        delta = left - right
        expected_pairs.append(
            {
                "family": key[0],
                "left": row["left"],
                "right": row["right"],
                "metric": key[1],
                **independent_interval(delta, reference["indices"]),
                "negative_improvement": int(np.count_nonzero(delta < 0)),
                "exact_tie": int(np.count_nonzero(delta == 0)),
                "positive_worsening": int(np.count_nonzero(delta > 0)),
                "nonworse": int(np.count_nonzero(delta <= 0)),
            }
        )
    required_pairs = {
        ("candidate_B", "energy"),
        ("candidate_B", "constant"),
        ("candidate_B", "latest_stage"),
        ("hybrid", "latest_stage"),
        ("constant", "latest_stage"),
    }
    available_pairs = {
        (r["left"], r["right"])
        for r in pairs
        if r["family"] == "primary" and r["metric"] == "normalized_regret"
    }
    if not required_pairs <= available_pairs:
        raise ValueError("Required primary paired comparisons missing")
    comparison.table(
        "paired_comparisons", pairs, expected_pairs, ("family", "left", "right", "metric")
    )
    class_differences, leave_one_out = [], []
    for row in expected_pairs:
        key = row["family"], row["metric"]
        delta = (
            reference["class_means"][key[0], row["left"], key[1]]
            - reference["class_means"][key[0], row["right"], key[1]]
        )
        for index, class_id in enumerate(reference["classes"]):
            identity = {k: row[k] for k in ("family", "left", "right", "metric")}
            identity.update(class_index=index, class_id=class_id)
            class_differences.append({**identity, "difference": float(delta[index])})
            without = float(np.mean(np.delete(delta, index)))
            full = float(np.mean(delta))
            leave_one_out.append(
                {
                    **identity,
                    "full_point": full,
                    "leave_one_class_out_point": without,
                    "change_from_full": without - full,
                }
            )
    class_keys = ("family", "left", "right", "metric", "class_id")
    comparison.table(
        "class_differences", _rows(audit / "class_differences.csv"), class_differences, class_keys
    )
    comparison.table(
        "leave_one_class_out", _rows(audit / "leave_one_class_out.csv"), leave_one_out, class_keys
    )
    choice_lookup = {
        (r["class_id"], r["seed"], r["set_id"], r["policy"]): r for r in reference["decisions"]
    }
    units = [
        (c, s, set_id)
        for c in reference["classes"]
        for s in reference["seeds"]
        for family, set_id in sorted(reference["sets"])
        if family == "primary"
    ]
    alias_pairs = [
        ("latest_stage", "stage_position_only"),
        ("latest_stage", "pure_G"),
        ("candidate_B", "hybrid"),
        ("candidate_B", "constant"),
    ]
    aliases = []
    for left_name, right_name in alias_pairs:
        same = sum(
            choice_lookup[*unit, left_name]["selected"]
            == choice_lookup[*unit, right_name]["selected"]
            for unit in units
        )
        latest15, pure15 = 0, 0
        for family, set_id in sorted(reference["sets"]):
            if family != "primary":
                continue
            sample = reference["classes"][0], reference["seeds"][0], set_id
            latest15 += choice_lookup[*sample, "latest_stage"]["selected_stage"] == 15
            pure15 += choice_lookup[*sample, "pure_G"]["selected_stage"] == 15
        aliases.append(
            dict(
                family="primary",
                left=left_name,
                right=right_name,
                units=len(units),
                same_selection=same,
                different_selection=len(units) - same,
                exact_selection_alias=same == len(units),
                latest_stage15_sets=latest15,
                pure_G_argmin_stage15_sets=pure15,
            )
        )
    comparison.table(
        "policy_equivalence",
        _rows(audit / "policy_equivalence.csv"),
        aliases,
        ("family", "left", "right"),
    )
    subset_ids = sorted(
        row["set_id"]
        for row in reference["headroom"]
        if row["family"] == "primary" and row["majority_rate"] < 0.75
    )
    subset_family = "exploratory_oracle_majority_lt_0.75"
    subset_expected = []
    for row in _rows(audit / "exploratory_subset_results.csv"):
        values = independent_class_means(
            reference["decisions"],
            reference["classes"],
            reference["seeds"],
            subset_ids,
            row["policy"],
            row["metric"],
        )
        subset_expected.append(
            dict(
                family=subset_family,
                policy=row["policy"],
                metric=row["metric"],
                **independent_interval(values, reference["indices"]),
            )
        )
    comparison.table(
        "exploratory_subset_results",
        _rows(audit / "exploratory_subset_results.csv"),
        subset_expected,
        ("family", "policy", "metric"),
    )
    subset_pairs = []
    for row in _rows(audit / "exploratory_subset_comparisons.csv"):
        left = independent_class_means(
            reference["decisions"],
            reference["classes"],
            reference["seeds"],
            subset_ids,
            row["left"],
            row["metric"],
        )
        right = independent_class_means(
            reference["decisions"],
            reference["classes"],
            reference["seeds"],
            subset_ids,
            row["right"],
            row["metric"],
        )
        delta = left - right
        subset_pairs.append(
            dict(
                family=subset_family,
                left=row["left"],
                right=row["right"],
                metric=row["metric"],
                **independent_interval(delta, reference["indices"]),
                negative_improvement=int(np.count_nonzero(delta < 0)),
                exact_tie=int(np.count_nonzero(delta == 0)),
                positive_worsening=int(np.count_nonzero(delta > 0)),
                nonworse=int(np.count_nonzero(delta <= 0)),
            )
        )
    comparison.table(
        "exploratory_subset_comparisons",
        _rows(audit / "exploratory_subset_comparisons.csv"),
        subset_pairs,
        ("family", "left", "right", "metric"),
    )
    tail_rows, selection_rows, oracle_rows = [], [], []
    primary_ids = sorted(set_id for family, set_id in reference["sets"] if family == "primary")
    for set_id in ["ALL", *primary_ids]:
        for policy in POLICIES:
            part = [
                r
                for r in reference["decisions"]
                if r["policy"] == policy and (set_id == "ALL" or r["set_id"] == set_id)
            ]
            for metric in ("absolute_regret", "normalized_regret"):
                raw = [r[metric] for r in part if r[metric] is not None]
                quantiles = np.quantile(raw, [0.5, 0.9, 0.95, 0.99, 1], method="linear")
                tail_rows.append(
                    dict(
                        family="primary",
                        set_id=set_id,
                        policy=policy,
                        metric=metric,
                        units=len(part),
                        valid_units=len(raw),
                        mean=float(np.mean(raw)),
                        **dict(
                            zip(
                                ("median", "Q90", "Q95", "Q99", "max"),
                                map(float, quantiles),
                                strict=True,
                            )
                        ),
                    )
                )
            if set_id == "ALL":
                continue
            for member in reference["sets"]["primary", set_id]:
                candidate = member["cell_id"]
                count = math.fsum(r["selected_probability_json"].get(candidate, 0.0) for r in part)
                selection_rows.append(
                    dict(
                        family="primary",
                        set_id=set_id,
                        policy=policy,
                        candidate=candidate,
                        stage=member["stage"],
                        count=count,
                        total=len(part),
                        frequency=count / len(part),
                    )
                )
                if policy == "constant":
                    count_oracle = sum(r["oracle"] == candidate for r in part)
                    oracle_rows.append(
                        dict(
                            family="primary",
                            set_id=set_id,
                            policy="oracle",
                            candidate=candidate,
                            stage=member["stage"],
                            count=count_oracle,
                            total=len(part),
                            frequency=count_oracle / len(part),
                        )
                    )
    comparison.table(
        "policy_tails",
        _rows(audit / "policy_tails.csv"),
        tail_rows,
        ("family", "set_id", "policy", "metric"),
    )
    comparison.table(
        "policy_selection_frequency",
        _rows(audit / "policy_selection_frequency.csv"),
        selection_rows,
        ("family", "set_id", "policy", "candidate"),
    )
    comparison.table(
        "oracle_frequency",
        _rows(audit / "oracle_frequency.csv"),
        oracle_rows,
        ("family", "set_id", "policy", "candidate"),
    )
    # Historical public summaries are independent comparators, not input to new calculations.
    original_summary = [
        r
        for r in _rows(root / "evidence/e1pred2r2/matched_cost_regret_summary.csv")
        if r["family"] == "primary"
    ]
    lookup = {(r["family"], r["policy"], r["metric"]): r for r in reference["summaries"]}
    for row in original_summary:
        key = row["family"], row["predictor"], row["metric"]
        for field in ("point", "low", "high", "lower95", "upper95"):
            comparison.check("historical_summary", key, field, row[field], lookup[key][field])
    # No secondary membership is present publicly; never create synthetic row-level evidence.
    secondary_source = [
        r
        for r in _rows(root / "evidence/e1pred2r2/predictor_pairwise_comparison.csv")
        if r["family"] != "primary"
    ]
    secondary_expected = []
    for record_type, relative in (
        ("summary", "evidence/e1pred2r2/matched_cost_regret_summary.csv"),
        ("paired_comparison", "evidence/e1pred2r2/predictor_pairwise_comparison.csv"),
    ):
        for row in _rows(root / relative):
            if row["family"] == "primary":
                continue
            secondary_expected.append(
                {
                    "predictor": "",
                    "left": "",
                    "right": "",
                    **row,
                    "record_type": record_type,
                    "analysis_status": "source_summary_only",
                    "source_path": relative,
                    "source_sha256": _hash(root / relative),
                }
            )
    comparison.table(
        "secondary_results",
        _rows(audit / "secondary_results.csv"),
        secondary_expected,
        ("family", "record_type", "predictor", "left", "right", "metric"),
        fields=(
            "point",
            "low",
            "high",
            "lower95",
            "upper95",
            "analysis_status",
            "source_path",
            "source_sha256",
        ),
    )
    output = dict(
        status="PASS" if not comparison.failures else "FAIL",
        scope="Independent public-scalar reconstruction; not raw tensor/model/leakage reproduction",
        imports_main_analysis=False,
        shared_reader_or_schema_code=False,
        implementation_inventory=[
            {"path": name, "bytes": (root / name).stat().st_size, "sha256": _hash(root / name)}
            for name in ("src/rcfs_dq/baseline_independent.py", "scripts/verify_baseline_audit.py")
        ],
        prior_inventory_files=reference["inventory_files"],
        audit_source_inventory_files=input_count,
        audit_output_inventory_files=output_count,
        observation_rows=len(reference["classes"])
        * len(reference["seeds"])
        * len(reference["candidates"]),
        decision_rows=len(reference["decisions"]),
        checked_scalar_fields=sum(comparison.counts.values()),
        manifest_semantic_checks=manifest_checks,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_indices_sha256=reference["bootstrap_indices_sha256"],
        tolerance={"atol": ATOL, "rtol": RTOL},
        source_score_max_abs_error=reference["source_score_max_abs_error"],
        secondary_status="source_summary_only" if secondary_source else "not_available",
        secondary_scope="Exact readback of historical summaries; no new secondary decisions",
        exploratory_subset_sets=len(subset_ids),
        primary_model_calls=0,
        failures=comparison.failures,
    )
    rows = [
        {**record, "checked_rows": comparison.counts[key]}
        for key, record in sorted(comparison.maximum.items())
    ]
    return output, rows
