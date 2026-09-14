"""Retrospective, scalar-only separation of stage weighting and source geometry.

No fitting or model import. The original frozen coefficients and evidence are inputs,
never outputs. The historical source certificate is a read-only provenance receipt,
not a claim to reproduce original tensors or tangent checks from public scalars.
"""

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .decisions import class_block_ci, evaluate_regret, select

CERTIFICATE_SHA256 = "f18b68a489ec71bc4194cfbac5d361aba26c8b2f1d2fe871706e51122068d6d8"
BOOTSTRAP_SEED = 2026091101
POLICIES = (
    "energy",
    "constant",
    "latest_stage",
    "candidate_B",
    "stage_only",
    "aligned_actual_source",
    "aligned_donor_source",
    "aligned_isotropic_source",
)
COMPARISONS = (
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stage_coefficients(candidates, geometry, beta):
    """Equal-source-cell log mean, including all bits/groups; no test outcome argument."""
    candidates = tuple(candidates)
    if not candidates or len({c.cell_id for c in candidates}) != len(candidates):
        raise ValueError("Unique, nonempty source candidates required")
    if set(geometry) != {c.cell_id for c in candidates} or not math.isfinite(beta):
        raise ValueError("Exact source geometry coverage and finite beta required")
    if any(not math.isfinite(v) or v <= 0 for v in geometry.values()):
        raise ValueError("Positive finite source gains required")
    by_stage = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: c.cell_id):
        by_stage[candidate.stage].append(math.log(geometry[candidate.cell_id]))
    result = {}
    for stage, logs in sorted(by_stage.items()):
        log_coefficient = beta * float(np.mean(logs))
        result[stage] = dict(
            stage=stage,
            source_cell_count=len(logs),
            mean_log_G=float(np.mean(logs)),
            log_coefficient=log_coefficient,
            coefficient=math.exp(log_coefficient),
        )
    return result


def geometry_scores(members, energies, geometry, beta):
    if len(members) != len({c.cell_id for c in members}):
        raise ValueError("Unique candidates required")
    if set(energies) != {c.cell_id for c in members}:
        raise ValueError("Exact energy coverage required")
    if any(not math.isfinite(e) or e < 0 for e in energies.values()):
        raise ValueError("Finite nonnegative energy required")
    if not math.isfinite(beta) or any(
        c.cell_id not in geometry
        or not math.isfinite(geometry[c.cell_id])
        or geometry[c.cell_id] <= 0
        for c in members
    ):
        raise ValueError("Positive finite source gains and finite beta required")
    scores = {c.cell_id: energies[c.cell_id] * geometry[c.cell_id] ** beta for c in members}
    if any(not math.isfinite(s) for s in scores.values()):
        raise ValueError("Nonfinite geometry score")
    return scores


def aligned_geometry(root, candidates, frozen):
    """Check pinned certificate and public evidence before accepting aligned gains."""
    root = Path(root)
    certificate_path = root / "analysis/pairing_ablation/source_alignment_certificate.json"
    if sha256(certificate_path) != CERTIFICATE_SHA256:
        raise ValueError("Historical source certificate hash mismatch")
    certificate = json.loads(certificate_path.read_text())
    source = root / "evidence/e1d1/directional_contrasts.csv"
    lock = next(
        r
        for r in certificate["input_source_inventory"]
        if r["source_path"].endswith("e1d1_directional_contrasts.csv")
    )
    if sha256(source) != lock["sha256"]:
        raise ValueError("Historical directional source hash mismatch")
    records = [r for r in read_csv(source) if r["alpha_role"] == "fixed_alpha"]
    by_cell = {r["cell_id"]: r for r in records}
    if len(by_cell) != len(records) or set(by_cell) != set(candidates):
        raise ValueError("Exact aligned historical cell coverage required")
    fields = {
        "aligned_actual_source": "LCG_actual",
        "aligned_donor_source": "LCG_shuffled",
        "aligned_isotropic_source": "LCG_iso",
    }
    gains = {
        name: {c: float(r[field]) for c, r in by_cell.items()} for name, field in fields.items()
    }
    checks = {r["cell_id"]: r for r in certificate["cells"]}
    alignment = []
    for cell_id, r in by_cell.items():
        c = candidates[cell_id]
        if c.stage != int(r["stage_index"]) or c.bits != int(r["weight_bits"]):
            raise ValueError("Historical taxonomy mismatch")
        if r["cell_tangent_status"] != "DISCOVERY_TANGENT_SUPPORTED":
            raise ValueError("Historical tangent-support mismatch")
        actual = gains["aligned_actual_source"][cell_id]
        for family in gains.values():
            if not math.isfinite(family[cell_id]) or family[cell_id] <= 0:
                raise ValueError("Positive finite aligned source gain required")
        if abs(actual - frozen["G"][cell_id]) > 1e-12 + 1e-10 * abs(actual):
            raise ValueError("Frozen actual source mismatch")
        alignment.append(
            dict(
                cell_id=cell_id,
                stage=c.stage,
                bits=c.bits,
                group=c.group,
                fixed_alpha=float(r["fixed_alpha"]),
                status="ALIGNED_HISTORICAL_TRIPLE",
                **{
                    k: checks[cell_id][k]
                    for k in (
                        "common_valid_pairs",
                        "represented_classes",
                        "common_receiver_sha256",
                        "same_receiver_sets",
                        "same_alpha",
                        "no_self_donor",
                    )
                },
                frozen_G=frozen["G"][cell_id],
                historical_actual_G=actual,
                historical_donor_G=gains["aligned_donor_source"][cell_id],
                historical_isotropic_G=gains["aligned_isotropic_source"][cell_id],
                frozen_actual_bitwise_equal=actual == frozen["G"][cell_id],
                frozen_actual_abs_difference=abs(actual - frozen["G"][cell_id]),
                source_sha256=lock["sha256"],
            )
        )
    return gains, alignment, certificate


def summarize(rows, classes, seeds, set_ids, indices):
    """Equal set -> equal seed -> equal class means; all undefined rows retained."""
    cube = {(r["class_id"], r["seed"], r["set_id"], r["policy"]): r for r in rows}
    if len(cube) != len(rows):
        raise ValueError("Duplicate decision row")
    summaries, class_rows, means = [], [], {}
    for policy in POLICIES:
        part = [r for r in rows if r["policy"] == policy]
        for metric in ("absolute_regret", "normalized_regret", "oracle_hit"):
            values = []
            for cls in classes:
                per_seed = []
                for seed in seeds:
                    samples = [cube[(cls, seed, s, policy)][metric] for s in set_ids]
                    per_seed.append(None if None in samples else float(np.mean(samples)))
                value = None if None in per_seed else float(np.mean(per_seed))
                class_rows.append(dict(class_id=cls, policy=policy, metric=metric, value=value))
                values.append(value)
            complete = None not in values
            means[(policy, metric)] = np.asarray(values, dtype=float) if complete else None
            raw = [r[metric] for r in part if r[metric] is not None]
            result = dict(
                policy=policy,
                family="primary",
                metric=metric,
                estimand="equal_set_seed_class_mean",
                unit_count=len(part),
                defined_units=len(raw),
                class_count=len(classes),
                complete=complete,
                source="RETROSPECTIVE_SCALAR_REANALYSIS",
            )
            if complete:
                result.update(class_block_ci(values, indices=indices))
            if raw:
                result.update(
                    zip(
                        ("median", "q90", "q95", "q99", "maximum"),
                        map(float, np.quantile(raw, [0.5, 0.9, 0.95, 0.99, 1], method="linear")),
                        strict=True,
                    )
                )
            summaries.append(result)
    comparisons, changes = [], []
    for left, right in COMPARISONS:
        changed = helped = hurt = equal_risk = 0
        for cls in classes:
            for seed in seeds:
                for set_id in set_ids:
                    lrow, rrow = cube[(cls, seed, set_id, left)], cube[(cls, seed, set_id, right)]
                    change = lrow["selected"] != rrow["selected"]
                    delta = lrow["absolute_regret"] - rrow["absolute_regret"]
                    changed += change
                    helped += change and delta < 0
                    hurt += change and delta > 0
                    equal_risk += change and delta == 0
                    if change:
                        changes.append(
                            dict(
                                class_id=cls,
                                seed=seed,
                                set_id=set_id,
                                left=left,
                                right=right,
                                selected_left=lrow["selected"],
                                selected_right=rrow["selected"],
                                absolute_difference=delta,
                                helped=delta < 0,
                                hurt=delta > 0,
                            )
                        )
        for metric in ("absolute_regret", "normalized_regret"):
            lval, rval = means[(left, metric)], means[(right, metric)]
            record = dict(
                left=left,
                right=right,
                metric=metric,
                family="primary",
                complete=lval is not None and rval is not None,
                changed_units=changed,
                helped_units=helped,
                hurt_units=hurt,
                changed_equal_risk_units=equal_risk,
                unit_count=len(classes) * len(seeds) * len(set_ids),
                negative_means="left_improves",
            )
            if record["complete"]:
                delta = lval - rval
                record.update(class_block_ci(delta, indices=indices))
                record.update(
                    improving_classes=int(np.sum(delta < 0)),
                    tied_classes=int(np.sum(delta == 0)),
                    worse_classes=int(np.sum(delta > 0)),
                )
            comparisons.append(record)
    return summaries, class_rows, comparisons, changes


def csv_bytes(rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def descriptive_profiles(data, class_rows, changes):
    """Separate source-geometry and test-risk profiles; no cross-population identity."""
    endpoints = {(r["class_id"], r["policy"], r["metric"]): r["value"] for r in class_rows}
    influence = []
    for left, right in COMPARISONS:
        for metric in ("absolute_regret", "normalized_regret"):
            pairs = [
                (c, endpoints[c, left, metric], endpoints[c, right, metric])
                for c in data["classes"]
            ]
            if any(
                left_value is None or right_value is None for _, left_value, right_value in pairs
            ):
                continue
            differences = np.array(
                [left_value - right_value for _, left_value, right_value in pairs]
            )
            full = float(np.mean(differences))
            for index, (cls, _, _) in enumerate(pairs):
                without = float(np.mean(np.delete(differences, index)))
                influence.append(
                    dict(
                        left=left,
                        right=right,
                        metric=metric,
                        class_id=cls,
                        class_difference=float(differences[index]),
                        full_point=full,
                        leave_one_class_out_point=without,
                        point_change=without - full,
                        estimand="descriptive_influence_only_no_class_removed_from_primary",
                    )
                )
    extremes = []
    for left, right in COMPARISONS:
        part = [r for r in changes if r["left"] == left and r["right"] == right]
        for order, label in ((False, "largest_improvements"), (True, "largest_losses")):
            selected = sorted(
                part,
                key=lambda r: (r["absolute_difference"], r["class_id"], r["seed"], r["set_id"]),
                reverse=order,
            )[:20]
            extremes.extend(dict(r, tail=label, rank=i + 1) for i, r in enumerate(selected))
    return influence, extremes


def run_audit(root):
    # Input validation is shared with P0; scientific scoring/aggregation is separate.
    from .baseline_audit import load_inputs

    root = Path(root)
    data = load_inputs(root)
    candidates, frozen = data["candidates"], data["frozen"]
    geometry, alignment, certificate = aligned_geometry(root, candidates, frozen)
    coefficients = stage_coefficients(candidates.values(), frozen["G"], frozen["beta"])
    geometry["candidate_B"] = frozen["G"]
    geometry["stage_only"] = {
        c.cell_id: math.exp(coefficients[c.stage]["mean_log_G"]) for c in candidates.values()
    }
    set_ids = sorted(s for s in data["sets"] if s.startswith("primary_"))
    rows, score_rows = [], []
    for (cls, seed), observations in sorted(data["observations"].items()):
        for set_id in set_ids:
            members = data["sets"][set_id]
            energies = {c.cell_id: observations[c.cell_id]["energy"] for c in members}
            risks = {c.cell_id: observations[c.cell_id]["native_Y"] for c in members}
            for policy in POLICIES:
                if policy == "energy":
                    selected = select(members, energies)
                elif policy == "constant":
                    selected = frozen["constants"][set_id]
                elif policy == "latest_stage":
                    selected = min(members, key=lambda c: (-c.stage, *c.tie_key)).cell_id
                else:
                    scores = geometry_scores(members, energies, geometry[policy], frozen["beta"])
                    if policy == "stage_only":
                        direct_scores = {
                            c.cell_id: energies[c.cell_id] * coefficients[c.stage]["coefficient"]
                            for c in members
                        }
                        for cell_id, direct in direct_scores.items():
                            if abs(scores[cell_id] - direct) > 1e-12 + 1e-10 * abs(direct):
                                raise ValueError("Stage coefficient score identity mismatch")
                        scores = direct_scores
                    selected = select(members, scores)
                    for c in members:
                        score_rows.append(
                            dict(
                                class_id=cls,
                                seed=seed,
                                set_id=set_id,
                                cell_id=c.cell_id,
                                policy=policy,
                                energy=energies[c.cell_id],
                                source_G=geometry[policy][c.cell_id],
                                score=scores[c.cell_id],
                            )
                        )
                result = evaluate_regret(members, risks, selected)
                rows.append(
                    dict(
                        class_id=cls,
                        seed=seed,
                        set_id=set_id,
                        family="primary",
                        policy=policy,
                        selected=selected,
                        selected_Y=risks[selected],
                        oracle_Y=risks[result["oracle"]],
                        oracle_hit=int(selected == result["oracle"]),
                        **result,
                    )
                )
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(data["classes"]), size=(10000, len(data["classes"]))
    )
    summaries, classes, comparisons, changes = summarize(
        rows, data["classes"], data["seeds"], set_ids, indices
    )
    influence, extremes = descriptive_profiles(data, classes, changes)
    geometry_records = [
        r
        for r in read_csv(root / "evidence/e1d1/directional_contrasts.csv")
        if r["alpha_role"] == "fixed_alpha"
    ]
    stage_profiles = []
    for stage in sorted(coefficients):
        source = [r for r in geometry_records if int(r["stage_index"]) == stage]
        native = [
            r
            for observations in data["observations"].values()
            for cell, r in observations.items()
            if candidates[cell].stage == stage
        ]
        stage_profiles.append(
            dict(
                stage=stage,
                geometry_source="E1D1_8classes_3seeds",
                native_risk_source="E1Pred2R2_16classes_4seeds",
                source_cells=len(source),
                geometry_remaining_updates=20 - stage - 1,
                native_intervention_suffix_updates=20 - stage,
                mean_source_G_actual=float(np.mean([float(r["LCG_actual"]) for r in source])),
                mean_source_G_donor=float(np.mean([float(r["LCG_shuffled"]) for r in source])),
                mean_source_G_iso=float(np.mean([float(r["LCG_iso"]) for r in source])),
                mean_source_L_state=float(np.mean([float(r["L-EDG-state"]) for r in source])),
                mean_test_energy=float(np.mean([r["energy"] for r in native])),
                mean_test_native_Y=float(np.mean([r["native_Y"] for r in native])),
                test_candidate_rows=len(native),
                aggregation="separate_equal_cell_source_and_test_means",
                pairing_information_bits_estimated=False,
            )
        )
    outputs = {
        "source_alignment.csv": alignment,
        "stage_coefficients.csv": list(coefficients.values()),
        "score_results.csv": summaries,
        "paired_comparisons.csv": comparisons,
        "class_endpoints.csv": classes,
        "decisions.csv": rows,
        "predictor_scores.csv": score_rows,
        "decision_changes.csv": changes,
        "leave_one_class_out.csv": influence,
        "largest_policy_changes.csv": extremes,
        "stage_observables.csv": stage_profiles,
    }
    missing = dict(
        status="COMPLETE_ALIGNED_HISTORICAL_ABLATION",
        missing_required_inputs=[],
        future_limitations=[
            "No independent target geometry",
            "No stage-neutral confirmation",
            "No raw-tensor regeneration or new model calls in this audit",
        ],
    )
    inputs = {r["path"]: r for r in data["source_inventory"]}
    # Record relative paths only; local repository paths never enter public artifacts.
    for p in (
        "evidence/e1d1/directional_contrasts.csv",
        "analysis/pairing_ablation/source_alignment_certificate.json",
    ):
        record = dict(path=p, sha256=sha256(root / p), bytes=(root / p).stat().st_size)
        if p in inputs and any(inputs[p][k] != record[k] for k in ("sha256", "bytes")):
            raise ValueError("Conflicting duplicate source inventory")
        inputs.setdefault(p, record)
    manifest = dict(
        schema_version=1,
        analysis="P1_RETROSPECTIVE_PAIRING_ABLATION",
        source_public_commit=data["public_commit"],
        input_inventory=list(inputs.values()),
        source_certificate_sha256=CERTIFICATE_SHA256,
        model_calls=0,
        fitted_parameters=0,
        source_coefficients_modified=0,
        class_count=len(data["classes"]),
        seed_count=len(data["seeds"]),
        primary_sets=len(set_ids),
        primary_units=len(rows) // len(POLICIES),
        policies=list(POLICIES),
        historical_source_status=certificate["status"],
        stage_coefficient="exp(beta * equal-source-cell mean(log frozen G)) by stage",
        all_source_cells_equal_weight=True,
        retrospective=True,
        bootstrap=dict(
            seed=BOOTSTRAP_SEED,
            generator="PCG64",
            draws=10000,
            unit="class",
            quantile_method="linear",
            indices_sha256=hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
        ),
        independent_verification="See independent_verification.json; separately implemented",
        limitations=[
            "Retrospective transfer of existing source geometry to an existing test panel",
            "Not an independent repeat of model outputs, support or tangent validity",
            "Stage-only coefficient is newly specified, not historically preregistered",
            "No causal attribution or mutual-information estimate",
        ],
    )
    payloads = {name: csv_bytes(table) for name, table in outputs.items()}
    payloads["missing_data_requirements.json"] = (json.dumps(missing, indent=2) + "\n").encode()
    for name in ("source_alignment_certificate.json",):
        payloads[name] = (root / "analysis/pairing_ablation" / name).read_bytes()
    return payloads, manifest


def export_audit(root, destination, *, check=False):
    """Never write evidence or arbitrary nonempty directories; check is read-only."""
    root = Path(root).resolve()
    raw = Path(destination).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("Symlink output locations are not permitted")
    destination = raw.resolve()
    canonical = root / "analysis/pairing_ablation"
    if destination == root or root.is_relative_to(destination):
        raise ValueError("Output cannot be the repository or its ancestor")
    if destination.is_relative_to(root) and destination != canonical:
        raise ValueError("In-repository output must use analysis/pairing_ablation")
    if (
        not check
        and destination != canonical
        and destination.exists()
        and any(destination.iterdir())
    ):
        raise ValueError("External output directory must be new or empty")
    payloads, manifest = run_audit(root)
    manifest["outputs"] = [
        dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()
    ]
    manifest["implementation"] = [
        dict(path=name, bytes=(root / name).stat().st_size, sha256=sha256(root / name))
        for name in ("src/rcfs_dq/pairing_audit.py", "scripts/audit_pairing_ablation.py")
    ]
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    files = {**payloads, "audit_manifest.json": manifest_bytes}
    for name in files:
        if (destination / name).is_symlink():
            raise ValueError("Symlink output file is not permitted")
    if check:
        for name, payload in files.items():
            if (destination / name).read_bytes() != payload:
                raise AssertionError(f"Recomputed P1 output mismatch: {name}")
    else:
        # Canonical updates require the previous generated outputs to remain intact.
        old_manifest = destination / "audit_manifest.json"
        if destination == canonical and old_manifest.exists():
            old = json.loads(old_manifest.read_text())
            if old.get("analysis") != "P1_RETROSPECTIVE_PAIRING_ABLATION":
                raise ValueError("Unrecognized canonical P1 output")
            for item in old["outputs"]:
                name = Path(item["path"]).name
                if name not in files or sha256(destination / name) != item["sha256"]:
                    raise ValueError(
                        "Modified/unrecognized canonical P1 output; preserve and review"
                    )
        known = (
            {Path(item["path"]).name for item in old["outputs"]}
            if (destination == canonical and old_manifest.exists())
            else set()
        )
        for name, payload in payloads.items():
            path = destination / name
            if path.exists() and name not in known and path.read_bytes() != payload:
                raise ValueError("Unrecognized existing P1 output must not be overwritten")
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (destination / name).write_bytes(payload)
    return dict(
        status="PASS_READONLY" if check else "COMPLETE",
        primary_units=manifest["primary_units"],
        model_calls=0,
        historical_source_status=manifest["historical_source_status"],
    )
