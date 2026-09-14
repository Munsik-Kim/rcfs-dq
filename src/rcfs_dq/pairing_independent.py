"""Independent scalar-only reference for stage/actual/donor/isotropic ablations.

Shared code is exclusively the separate independent P0 IO, scalar reference, and
comparison implementation.  No main scoring/selection/aggregation is imported.
The historical provenance certificate is hash-checked, not a substitute for an
independent re-execution of original receiver probes or tangent-validity tests.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .baseline_independent import (
    ATOL,
    RTOL,
    Comparison,
    _finite,
    _hash,
    _rows,
    independent_class_means,
    independent_evaluate,
    independent_interval,
    reference_from_sources,
    validate_inventory,
    validate_source_metadata,
)

PAIRING_POLICIES = (
    "energy",
    "constant",
    "latest_stage",
    "candidate_B",
    "stage_only",
    "aligned_actual_source",
    "aligned_donor_source",
    "aligned_isotropic_source",
)
PAIRING_COMPARISONS = (
    ("candidate_B", "stage_only"),
    ("aligned_actual_source", "stage_only"),
    ("aligned_actual_source", "aligned_donor_source"),
    ("aligned_donor_source", "aligned_isotropic_source"),
    ("aligned_actual_source", "aligned_isotropic_source"),
    *(
        (p, b)
        for p in (
            "stage_only",
            "aligned_actual_source",
            "aligned_donor_source",
            "aligned_isotropic_source",
        )
        for b in ("constant", "latest_stage")
    ),
)


def independent_stage_coefficients(candidates, gains, beta):
    """Equal-source-cell log means; coefficient construction cannot access test Y."""
    if set(gains) != set(candidates) or not math.isfinite(beta):
        raise ValueError("Stage coefficient source coverage or beta invalid")
    stages = sorted({row["stage"] for row in candidates.values()})
    output = []
    for stage in stages:
        names = sorted(name for name, row in candidates.items() if row["stage"] == stage)
        logs = np.log(np.asarray([_finite(gains[name], positive=True) for name in names]))
        mean = float(np.sum(logs) / len(logs))
        log_coefficient = beta * mean
        coefficient = math.exp(log_coefficient)
        _finite(coefficient, positive=True)
        output.append(
            dict(
                stage=stage,
                source_cell_count=len(names),
                mean_log_G=mean,
                log_coefficient=log_coefficient,
                coefficient=coefficient,
            )
        )
    return output


def pairing_reference(root):
    root = Path(root)
    source = reference_from_sources(root)
    frozen = json.loads((root / "configs/frozen_predictor.json").read_text())
    stages = independent_stage_coefficients(source["candidates"], frozen["G"], frozen["beta"])
    stage_values = {row["stage"]: row for row in stages}
    historical = [
        r
        for r in _rows(root / "evidence/e1d1/directional_contrasts.csv")
        if r["alpha_role"] == "fixed_alpha"
    ]
    history = {r["cell_id"]: r for r in historical}
    if len(history) != len(historical) or set(history) != set(source["candidates"]):
        raise ValueError("Historical geometry family is incomplete or duplicated")
    certificate = json.loads(
        (root / "analysis/pairing_ablation/source_alignment_certificate.json").read_text()
    )
    certified_cells = {row["cell_id"]: row for row in certificate["cells"]}
    if len(certified_cells) != len(certificate["cells"]) or set(certified_cells) != set(history):
        raise ValueError("Historical certificate cell coverage mismatch")
    geometry = {policy: {} for policy in PAIRING_POLICIES[3:]}
    for name, row in history.items():
        candidate = source["candidates"][name]
        if (
            row["group_name"] != candidate["group"]
            or int(row["stage_index"]) != candidate["stage"]
            or int(row["weight_bits"]) != candidate["bits"]
            or row["cell_tangent_status"] != "DISCOVERY_TANGENT_SUPPORTED"
            or int(row["common_valid_pair_count"]) != certified_cells[name]["common_valid_pairs"]
            or not 1 <= int(row["common_valid_pair_count"]) <= 24
            or int(row["represented_class_count"]) != 8
            or int(row["represented_class_count"]) != certified_cells[name]["represented_classes"]
        ):
            raise ValueError("Geometry source identity/support metadata mismatch")
        iso = (
            _finite(row["LCG_iso0"], positive=True) + _finite(row["LCG_iso1"], positive=True)
        ) / 2
        if abs(iso - float(row["LCG_iso"])) > ATOL + RTOL * abs(iso):
            raise ValueError("Isotropic aggregate differs from equal two-probe mean")
        geometry["candidate_B"][name] = _finite(frozen["G"][name], positive=True)
        geometry["aligned_actual_source"][name] = _finite(row["LCG_actual"], positive=True)
        geometry["aligned_donor_source"][name] = _finite(row["LCG_shuffled"], positive=True)
        geometry["aligned_isotropic_source"][name] = iso
        geometry["stage_only"][name] = math.exp(stage_values[candidate["stage"]]["mean_log_G"])
    observations = {
        (int(r["class_id"]), int(r["seed"]), r["cell_id"]): r
        for r in _rows(root / "evidence/e1pred2r2/candidate_observations.csv")
    }
    old_choices = {
        (r["class_id"], r["seed"], r["set_id"], r["policy"]): r["selected"]
        for r in source["decisions"]
    }
    decisions, score_rows = [], []
    for class_id in source["classes"]:
        for seed in source["seeds"]:
            for (family, set_id), members in sorted(source["sets"].items()):
                if family != "primary":
                    raise ValueError("P1 scope must not silently add secondary sets")
                order = [member["cell_id"] for member in members]
                y = {name: float(observations[class_id, seed, name]["native_Y"]) for name in order}
                energy = {
                    name: float(observations[class_id, seed, name]["energy"]) for name in order
                }
                for policy in PAIRING_POLICIES:
                    if policy in ("energy", "constant", "latest_stage"):
                        selected = old_choices[class_id, seed, set_id, policy]
                    else:
                        score = {}
                        for name in order:
                            value = energy[name] * geometry[policy][name] ** frozen["beta"]
                            if policy == "stage_only":
                                stage = source["candidates"][name]["stage"]
                                value = energy[name] * stage_values[stage]["coefficient"]
                            score[name] = _finite(value)
                            score_rows.append(
                                dict(
                                    class_id=class_id,
                                    seed=seed,
                                    set_id=set_id,
                                    cell_id=name,
                                    policy=policy,
                                    energy=energy[name],
                                    source_G=geometry[policy][name],
                                    score=value,
                                )
                            )
                        selected = order[int(np.argmin([score[name] for name in order]))]
                    value = independent_evaluate(order, y, selected)
                    decisions.append(
                        dict(
                            class_id=class_id,
                            seed=seed,
                            set_id=set_id,
                            family=family,
                            policy=policy,
                            **value,
                        )
                    )
    return {
        **source,
        "decisions": decisions,
        "predictor_scores": score_rows,
        "stage_coefficients": stages,
    }


def validate_pairing_contract(manifest, reference):
    """Require declared scope to agree with independently reconstructed scalar support."""
    set_ids = [s for family, s in reference["sets"] if family == "primary"]
    expected = {
        "schema_version": 1,
        "analysis": "P1_RETROSPECTIVE_PAIRING_ABLATION",
        "all_source_cells_equal_weight": True,
        "fitted_parameters": 0,
        "source_coefficients_modified": 0,
        "model_calls": 0,
        "retrospective": True,
        "class_count": len(reference["classes"]),
        "seed_count": len(reference["seeds"]),
        "primary_sets": len(set_ids),
        "primary_units": len(reference["classes"]) * len(reference["seeds"]) * len(set_ids),
        "policies": list(PAIRING_POLICIES),
        "stage_coefficient": "exp(beta * equal-source-cell mean(log frozen G)) by stage",
        "bootstrap": {
            "draws": 10000,
            "generator": "PCG64",
            "indices_sha256": reference["bootstrap_indices_sha256"],
            "quantile_method": "linear",
            "seed": 2026091101,
            "unit": "class",
        },
    }
    for field, value in expected.items():
        if field not in manifest or json.dumps(manifest[field], sort_keys=True) != json.dumps(
            value, sort_keys=True
        ):
            raise ValueError(f"P1 manifest semantic mismatch: {field}")
    return len(expected)


def verify_pairing_audit(root, audit_directory):
    root, audit = Path(root), Path(audit_directory)
    manifest = json.loads((audit / "audit_manifest.json").read_text())
    inputs = validate_inventory(root, manifest["input_inventory"])
    outputs = validate_inventory(audit, manifest["outputs"])
    certificate = audit / "source_alignment_certificate.json"
    if _hash(certificate) != manifest["source_certificate_sha256"]:
        raise ValueError("Historical geometry alignment certificate changed")
    reference = pairing_reference(root)
    semantic_checks = validate_pairing_contract(manifest, reference)
    public_records = [
        r for r in manifest["input_inventory"] if not r["path"].startswith("analysis/")
    ]
    public_source = validate_source_metadata(root, public_records, reference["source_inventory"])
    if manifest["source_public_commit"] != public_source:
        raise ValueError("P1 source commit differs from input provenance")
    comparison = Comparison()
    comparison.check(
        "manifest",
        "bootstrap",
        "indices_sha256",
        manifest["bootstrap"]["indices_sha256"],
        reference["bootstrap_indices_sha256"],
    )
    comparison.table(
        "stage_coefficients",
        _rows(audit / "stage_coefficients.csv"),
        reference["stage_coefficients"],
        ("stage",),
    )
    comparison.table(
        "predictor_scores",
        _rows(audit / "predictor_scores.csv"),
        reference["predictor_scores"],
        ("class_id", "seed", "set_id", "cell_id", "policy"),
    )
    decision_fields = (
        "selected",
        "oracle",
        "selected_Y",
        "oracle_Y",
        "oracle_hit",
        "absolute_regret",
        "normalized_regret",
        "resolved",
        "risk_range",
        "range_tau",
    )
    comparison.table(
        "decisions",
        _rows(audit / "decisions.csv"),
        reference["decisions"],
        ("family", "class_id", "seed", "set_id", "policy"),
        decision_fields,
    )
    set_ids = sorted(key[1] for key in reference["sets"])
    means, summary_rows, class_rows = {}, [], []
    for policy in PAIRING_POLICIES:
        for metric in ("absolute_regret", "normalized_regret", "oracle_hit"):
            values = independent_class_means(
                reference["decisions"],
                reference["classes"],
                reference["seeds"],
                set_ids,
                policy,
                metric,
            )
            if values is None:
                raise ValueError("Unresolved P1 scalar coverage")
            means[policy, metric] = values
            raw = [row[metric] for row in reference["decisions"] if row["policy"] == policy]
            tails = np.quantile(raw, [0.5, 0.9, 0.95, 0.99, 1], method="linear")
            summary_rows.append(
                dict(
                    family="primary",
                    policy=policy,
                    metric=metric,
                    **independent_interval(values, reference["indices"]),
                    **dict(
                        zip(
                            ("median", "q90", "q95", "q99", "maximum"),
                            map(float, tails),
                            strict=True,
                        )
                    ),
                    unit_count=len(raw),
                    defined_units=len(raw),
                    class_count=len(values),
                    complete=True,
                )
            )
            class_rows.extend(
                dict(class_id=c, policy=policy, metric=metric, value=float(v))
                for c, v in zip(reference["classes"], values, strict=True)
            )
    comparison.table(
        "score_results",
        _rows(audit / "score_results.csv"),
        summary_rows,
        ("family", "policy", "metric"),
    )
    comparison.table(
        "class_endpoints",
        _rows(audit / "class_endpoints.csv"),
        class_rows,
        ("class_id", "policy", "metric"),
    )
    lookup = {
        (r["class_id"], r["seed"], r["set_id"], r["policy"]): r for r in reference["decisions"]
    }
    comparisons, changes = [], []
    for left_name, right_name in PAIRING_COMPARISONS:
        changed = helped = hurt = tied_risk = 0
        for cls in reference["classes"]:
            for seed in reference["seeds"]:
                for set_id in set_ids:
                    left = lookup[cls, seed, set_id, left_name]
                    right = lookup[cls, seed, set_id, right_name]
                    if left["selected"] == right["selected"]:
                        continue
                    changed += 1
                    delta = left["absolute_regret"] - right["absolute_regret"]
                    helped += delta < 0
                    hurt += delta > 0
                    tied_risk += delta == 0
                    changes.append(
                        dict(
                            class_id=cls,
                            seed=seed,
                            set_id=set_id,
                            left=left_name,
                            right=right_name,
                            selected_left=left["selected"],
                            selected_right=right["selected"],
                            absolute_difference=delta,
                            helped=delta < 0,
                            hurt=delta > 0,
                        )
                    )
        for metric in ("absolute_regret", "normalized_regret"):
            difference = means[left_name, metric] - means[right_name, metric]
            comparisons.append(
                dict(
                    family="primary",
                    left=left_name,
                    right=right_name,
                    metric=metric,
                    **independent_interval(difference, reference["indices"]),
                    complete=True,
                    changed_units=changed,
                    helped_units=helped,
                    hurt_units=hurt,
                    changed_equal_risk_units=tied_risk,
                    unit_count=len(reference["classes"]) * len(reference["seeds"]) * len(set_ids),
                    improving_classes=int(np.count_nonzero(difference < 0)),
                    tied_classes=int(np.count_nonzero(difference == 0)),
                    worse_classes=int(np.count_nonzero(difference > 0)),
                )
            )
    comparison.table(
        "paired_comparisons",
        _rows(audit / "paired_comparisons.csv"),
        comparisons,
        ("family", "left", "right", "metric"),
    )
    comparison.table(
        "decision_changes",
        _rows(audit / "decision_changes.csv"),
        changes,
        ("class_id", "seed", "set_id", "left", "right"),
    )
    report = dict(
        status="PASS" if not comparison.failures else "FAIL",
        scope="Independent public-scalar P1 recomputation; historical certificate readback only",
        source_alignment_independently_reexecuted=False,
        original_receiver_tensors_independently_recomputed=False,
        source_certificate_sha256=_hash(certificate),
        imports_main_analysis=False,
        implementation_inventory=[
            {"path": name, "bytes": (root / name).stat().st_size, "sha256": _hash(root / name)}
            for name in (
                "src/rcfs_dq/baseline_independent.py",
                "src/rcfs_dq/pairing_independent.py",
                "scripts/verify_baseline_audit.py",
            )
        ],
        shared_code="Independent P0 reader, reference and comparison utilities only",
        audit_source_inventory_files=inputs,
        audit_output_inventory_files=outputs,
        predictor_score_rows=len(reference["predictor_scores"]),
        decision_rows=len(reference["decisions"]),
        checked_scalar_fields=sum(comparison.counts.values()),
        manifest_semantic_checks=semantic_checks,
        bootstrap_indices_sha256=hashlib.sha256(
            reference["indices"].astype("<i8").tobytes()
        ).hexdigest(),
        tolerance={"atol": ATOL, "rtol": RTOL},
        model_calls=0,
        failures=comparison.failures,
    )
    rows = [
        {**record, "checked_rows": comparison.counts[key]}
        for key, record in sorted(comparison.maximum.items())
    ]
    return report, rows
