"""Retrospective free-baseline audit of published scalar decisions.

No model imports, fitting, test-conditioned coefficients, or evidence writes.
This main implementation deliberately reuses the public frozen-score and
decision primitives. ``baseline_independent`` is the separate reference.
"""

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rcfs_dq.decisions import Candidate, class_block_ci, evaluate_regret, exact_cost_sets, select
from rcfs_dq.scoring import FrozenCandidateB
from rcfs_dq.verification import assert_close, file_sha256, verify_inventory

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
METRICS = ("absolute_regret", "normalized_regret", "oracle_hit")
COMPARISONS = (
    ("candidate_B", "energy"),
    ("candidate_B", "constant"),
    ("candidate_B", "latest_stage"),
    ("hybrid", "latest_stage"),
    ("constant", "latest_stage"),
    ("hybrid", "energy"),
    ("hybrid", "constant"),
)
BOOTSTRAP_SEED = 2026091101
BOOTSTRAP_DRAWS = 10000
PUBLIC_SOURCE_COMMIT = "af7b402839d1052ece994c80a91e59e10d7474dc"
SCOPE = "retrospective_public_scalar_reanalysis"
INPUT_PATHS = (
    "configs/candidates.csv",
    "configs/cost_sets.csv",
    "configs/frozen_predictor.json",
    "evidence/e1pred2r2/candidate_observations.csv",
    "evidence/e1pred2r2/matched_cost_regret_summary.csv",
    "evidence/e1pred2r2/predictor_pairwise_comparison.csv",
    "evidence/e1pred2r2/decision_diversity.csv",
    "evidence/e1d1/directional_contrasts.csv",
    "evidence/e1d1/group_stage_bit_summary.csv",
    "evidence/e1c/global_tests.csv",
    "evidence/e1c/directional_replication.csv",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicate CSV columns")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError("Malformed CSV row")
        return rows


def git_blob_sha1(path):
    """Git blob identity from verified bytes, also available in a Git-free export."""
    payload = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def _finite_nonnegative(value, label):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Finite nonnegative {label} required")
    return number


def load_inputs(root):
    """Fail closed on the protected inventory, exact grid, costs and frozen scores."""
    root = Path(root).resolve()
    manifest = json.loads((root / "PUBLIC_EVIDENCE_MANIFEST.json").read_text())
    inventory_count = verify_inventory(root, manifest)
    inventory = {row["path"]: row for row in manifest["files"]}
    if not set(INPUT_PATHS) <= set(inventory):
        raise ValueError("Required source missing from original evidence inventory")
    frozen = json.loads((root / "configs/frozen_predictor.json").read_text())
    if frozen["fit_allowed"] is not False:
        raise ValueError("Frozen predictor contract required")
    score = FrozenCandidateB(frozen["G"], frozen["beta"])
    candidates = {}
    groups = {}
    for row in read_csv(root / "configs/candidates.csv"):
        groups.setdefault(row["group"], len(groups))
        candidate = Candidate(
            row["cell_id"],
            row["group"],
            int(row["stage"]),
            int(row["bits"]),
            int(row["numel"]),
            groups[row["group"]],
            row["regime"],
        )
        shapes = json.loads(row["shapes"])
        if (
            candidate.cell_id in candidates
            or candidate.packed_bits != int(row["packed_bits"])
            or candidate.saving_bits != int(row["saving_bits"])
            or sum(math.prod(shape) for shape in shapes.values()) != candidate.numel
        ):
            raise ValueError("Candidate duplicate or shape/cost mismatch")
        candidates[candidate.cell_id] = candidate
    if len(candidates) != 81 or set(candidates) != set(frozen["G"]):
        raise ValueError("Exact Core81 / geometry coverage required")
    sets = defaultdict(list)
    for row in read_csv(root / "configs/cost_sets.csv"):
        candidate = candidates[row["cell_id"]]
        if (
            row["family"] != "primary"
            or int(row["candidate_count"]) != 3
            or candidate.saving_bits != int(row["saving_bits"])
            or candidate.packed_bits != int(row["packed_bits"])
        ):
            raise ValueError("Exact primary source membership/cost required")
        sets[row["set_id"]].append(candidate)
    if len(sets) != 27 or set(sets) != set(frozen["constants"]):
        raise ValueError("Exact primary sets and frozen constants required")
    for set_id, members in sets.items():
        if (
            len(members) != 3
            or len({c.cell_id for c in members}) != 3
            or frozen["constants"][set_id] not in {c.cell_id for c in members}
        ):
            raise ValueError("Duplicate/missing set member or invalid source Constant")
    if {tuple(sorted(c.cell_id for c in v)) for v in sets.values()} != {
        tuple(sorted(c.cell_id for c in v))
        for v in exact_cost_sets(list(candidates.values())).values()
    }:
        raise ValueError("Primary registry differs from exact group/bit/saving partition")
    sets = {key: tuple(sorted(value, key=lambda c: c.tie_key)) for key, value in sets.items()}
    observations = defaultdict(dict)
    score_errors = []
    for source_row in read_csv(root / "evidence/e1pred2r2/candidate_observations.csv"):
        row = dict(source_row)
        key = int(row["class_id"]), int(row["seed"])
        cell = row["cell_id"]
        if cell not in candidates or cell in observations[key]:
            raise ValueError("Duplicate or unknown observation candidate")
        for field in ("energy", "native_Y", "source_B_score"):
            row[field] = _finite_nonnegative(row[field], field)
        actual = score.score(cell, row["energy"])
        error = assert_close(actual, row["source_B_score"])
        score_errors.append(error)
        observations[key][cell] = row
    classes = sorted({key[0] for key in observations})
    seeds = sorted({key[1] for key in observations})
    if (
        len(classes) != 16
        or len(seeds) != 4
        or set(observations) != {(c, s) for c in classes for s in seeds}
        or any(set(rows) != set(candidates) for rows in observations.values())
    ):
        raise ValueError("Exact 16 class x 4 seed x 81 candidate coverage required")
    # Historical public evidence origin, not whichever checkout runs this audit.
    head = PUBLIC_SOURCE_COMMIT
    source_inventory = []
    for name in INPUT_PATHS:
        path = root / name
        record = dict(inventory[name])
        record.update(
            public_commit=head,
            public_blob=git_blob_sha1(path),
        )
        if path.suffix == ".csv":
            table = read_csv(path)
            record.update(schema=list(table[0]) if table else [], rows=len(table))
        source_inventory.append(record)
    return dict(
        candidates=candidates,
        sets=sets,
        observations=dict(observations),
        frozen=frozen,
        classes=classes,
        seeds=seeds,
        source_inventory=source_inventory,
        protected_inventory_count=inventory_count,
        source_base_commit=manifest["source_base_commit"],
        public_commit=head,
        maximum_source_score_abs_error=max(score_errors),
        public_inventory_sha256=file_sha256(root / "PUBLIC_EVIDENCE_MANIFEST.json"),
    )


def evaluate_policy(members, energies, frozen, set_id, policy):
    """Selection sees features and frozen configuration, never terminal risk."""
    if set(energies) != {c.cell_id for c in members}:
        raise ValueError("Exact energy coverage required")
    if len(members) != len({c.cell_id for c in members}):
        raise ValueError("Duplicate candidate")
    for energy in energies.values():
        _finite_nonnegative(energy, "energy")
    if policy == "uniform_random_expected":
        return None
    if policy == "constant":
        choice = frozen["constants"][set_id]
        if choice not in energies:
            raise ValueError("Source Constant outside set")
        return choice
    if policy in {"latest_stage", "stage_position_only"}:
        # Negative stages are legitimate ordering scores, not nonnegative risks.
        return min(members, key=lambda c: (-c.stage, *c.tie_key)).cell_id
    if policy == "energy":
        return select(members, energies)
    if policy == "pure_G":
        return select(members, {c.cell_id: frozen["G"][c.cell_id] for c in members})
    score = FrozenCandidateB(frozen["G"], frozen["beta"])
    scores = {c.cell_id: score.score(c.cell_id, energies[c.cell_id]) for c in members}
    if policy == "candidate_B":
        return select(members, scores)
    if policy == "hybrid":
        return select(
            members,
            {
                c.cell_id: energies[c.cell_id] if c.regime == "M3" else scores[c.cell_id]
                for c in members
            },
            hybrid=True,
        )
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_unit(members, values, selected):
    """Evaluate one sealed choice, including the exact uniform expectation."""
    if selected is not None:
        result = evaluate_regret(members, values, selected)
        result.update(
            selected_Y=values[selected],
            oracle_Y=values[result["oracle"]],
            oracle_hit=float(selected == result["oracle"]),
            all_minima_hit=float(values[selected] == values[result["oracle"]]),
        )
        return result
    # Validates finite/nonnegative coverage and exact costs through the source primitive.
    oracle = select(members, values)
    result = evaluate_regret(members, values, oracle)
    # Mean of nonnegative candidate regrets avoids cancellation for a tied set.
    expected_regret = float(np.mean([values[c.cell_id] - values[oracle] for c in members]))
    expected_y = values[oracle] + expected_regret
    result.update(
        selected_Y=expected_y,
        oracle_Y=values[oracle],
        absolute_regret=expected_regret,
        normalized_regret=expected_regret / result["risk_range"] if result["resolved"] else None,
        oracle_hit=1.0 / len(members),
        all_minima_hit=sum(v == values[oracle] for v in values.values()) / len(members),
    )
    return result


def build_decisions(data):
    decisions = []
    for (class_id, seed), records in sorted(data["observations"].items()):
        for set_id, members in sorted(data["sets"].items()):
            energies = {c.cell_id: records[c.cell_id]["energy"] for c in members}
            values = {c.cell_id: records[c.cell_id]["native_Y"] for c in members}
            for policy in POLICIES:
                selected = evaluate_policy(members, energies, data["frozen"], set_id, policy)
                result = evaluate_unit(members, values, selected)
                probabilities = (
                    {c.cell_id: 1.0 / len(members) for c in members}
                    if selected is None
                    else {selected: 1.0}
                )
                decisions.append(
                    dict(
                        family="primary",
                        class_id=class_id,
                        seed=seed,
                        set_id=set_id,
                        policy=policy,
                        selected=selected,
                        selected_stage=data["candidates"][selected].stage if selected else None,
                        oracle_stage=data["candidates"][result["oracle"]].stage,
                        candidate_count=len(members),
                        saving_bits=members[0].saving_bits,
                        selected_probability_json=json.dumps(probabilities, sort_keys=True),
                        unit="class_seed_exact_cost_set",
                        estimand=SCOPE,
                        **result,
                    )
                )
    return decisions


def summarize_decisions(rows, indices):
    """Equal sets -> equal seeds -> equal classes, with no silent missing support."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"], row["policy"]].append(row)
    summary, class_rows, means = [], [], {}
    for (family, policy), group in sorted(grouped.items()):
        classes = sorted({r["class_id"] for r in group})
        seeds = sorted({r["seed"] for r in group})
        sets = sorted({r["set_id"] for r in group})
        keys = [(r["class_id"], r["seed"], r["set_id"]) for r in group]
        expected = {(c, s, b) for c in classes for s in seeds for b in sets}
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("Incomplete or duplicate aggregation grid")
        indexed = {(r["class_id"], r["seed"], r["set_id"]): r for r in group}
        for metric in METRICS:
            values = []
            for c in classes:
                seed_values = []
                for s in seeds:
                    selected = [indexed[c, s, b][metric] for b in sets]
                    seed_values.append(
                        float(np.mean(selected)) if all(v is not None for v in selected) else None
                    )
                value = (
                    float(np.mean(seed_values)) if all(v is not None for v in seed_values) else None
                )
                values.append(value)
                class_rows.append(
                    dict(
                        family=family,
                        policy=policy,
                        metric=metric,
                        class_id=c,
                        value=value,
                        sets=len(sets),
                        seeds=len(seeds),
                        unit="equal_set_equal_seed_class_mean",
                        estimand=SCOPE,
                    )
                )
            raw = [r[metric] for r in group if r[metric] is not None]
            complete = all(v is not None for v in values)
            ci = (
                class_block_ci(values, indices=indices)
                if complete
                else dict.fromkeys(("point", "low", "high", "lower95", "upper95"))
            )
            if complete:
                means[family, policy, metric] = np.asarray(values)
            summary.append(
                dict(
                    family=family,
                    policy=policy,
                    metric=metric,
                    **ci,
                    classes=len(classes),
                    seeds=len(seeds),
                    sets=len(sets),
                    units=len(group),
                    valid_units=len(raw),
                    undefined_units=len(group) - len(raw),
                    aggregation="equal_set_mean_then_equal_seed_mean_then_equal_class_mean",
                    unit="class_block",
                    estimand=SCOPE,
                )
            )
    return summary, class_rows, means


def paired_comparisons(means, indices, comparisons=COMPARISONS):
    paired, class_diffs, influence = [], [], []
    families = sorted({key[0] for key in means})
    for family in families:
        for left, right in comparisons:
            for metric in METRICS:
                if (family, left, metric) not in means or (family, right, metric) not in means:
                    continue
                delta = means[family, left, metric] - means[family, right, metric]
                ci = class_block_ci(delta, indices=indices)
                paired.append(
                    dict(
                        family=family,
                        left=left,
                        right=right,
                        metric=metric,
                        **ci,
                        negative_improvement=int(np.sum(delta < 0)),
                        exact_tie=int(np.sum(delta == 0)),
                        positive_worsening=int(np.sum(delta > 0)),
                        nonworse=int(np.sum(delta <= 0)),
                        sign_convention="negative_improves_regret_positive_improves_oracle_hit",
                        estimand=SCOPE,
                        unit="paired_class_block",
                    )
                )
                for i, difference in enumerate(delta):
                    class_diffs.append(
                        dict(
                            family=family,
                            left=left,
                            right=right,
                            metric=metric,
                            class_index=i,
                            difference=float(difference),
                            estimand=SCOPE,
                            unit="paired_class_mean",
                        )
                    )
                    if len(delta) > 1:
                        omitted_mean = float(np.delete(delta, i).mean())
                        influence.append(
                            dict(
                                family=family,
                                left=left,
                                right=right,
                                metric=metric,
                                class_index=i,
                                full_point=ci["point"],
                                leave_one_class_out_point=omitted_mean,
                                change_from_full=omitted_mean - ci["point"],
                                estimand="retrospective_influence_no_class_deleted_from_main",
                            )
                        )
    return paired, class_diffs, influence


def empirical_headroom(risk_matrix, tie_order):
    """Post-outcome best fixed diagnostics; absolute and normalized optima differ."""
    values = np.asarray(risk_matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(tie_order)
        or values.shape[0] == 0
        or len(set(tie_order)) != len(tie_order)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Finite nonnegative balanced risk matrix and unique tie order required")
    oracle = np.argmin(values, axis=1)
    counts = np.bincount(oracle, minlength=len(tie_order))
    majority = int(np.argmax(counts))
    means = values.mean(axis=0)
    best = int(np.argmin(means))
    minima = values.min(axis=1)
    spans = values.max(axis=1) - minima
    valid = spans > 1e-12 * np.maximum(values.max(axis=1), 1.0)
    nr = (values[valid] - minima[valid, None]) / spans[valid, None]
    best_nr = int(np.argmin(nr.mean(axis=0))) if valid.all() else None
    frequencies = counts[counts > 0] / len(values)
    return dict(
        majority_candidate=tie_order[majority],
        majority_rate=float(counts[majority] / len(values)),
        oracle_entropy_bits=float(-np.sum(frequencies * np.log2(frequencies))),
        fully_fixed=bool(counts[majority] == len(values)),
        oracle_tie_units=int(np.sum(np.sum(values == minima[:, None], axis=1) > 1)),
        empirical_best_fixed_abs_candidate=tie_order[best],
        H_abs=float(means[best] - minima.mean()),
        empirical_best_fixed_nr_candidate=tie_order[best_nr] if best_nr is not None else None,
        H_normalized=float(nr[:, best_nr].mean()) if best_nr is not None else None,
        headroom_normalized_valid_units=int(valid.sum()),
        majority_equals_minimum_expected_risk=bool(majority == best),
    )


def headroom_rows(data):
    output, frequencies = [], []
    for set_id, members in sorted(data["sets"].items()):
        matrix = np.asarray(
            [
                [records[c.cell_id]["native_Y"] for c in members]
                for _, records in sorted(data["observations"].items())
            ]
        )
        ids = [c.cell_id for c in members]
        result = empirical_headroom(matrix, ids)
        counts = Counter(ids[i] for i in np.argmin(matrix, axis=1))
        output.append(
            dict(
                family="primary",
                set_id=set_id,
                group=members[0].group,
                bits=members[0].bits,
                saving_bits=members[0].saving_bits,
                units=len(matrix),
                **result,
                majority_stage=data["candidates"][result["majority_candidate"]].stage,
                eligible_exploratory_subset=result["majority_rate"] < 0.75,
                candidate_frequencies_json=json.dumps(counts, sort_keys=True),
                unit="exact_cost_set_equal_class_seed",
                estimand="test_best_fixed_diagnostic_not_baseline",
            )
        )
        for c in members:
            frequencies.append(
                dict(
                    family="primary",
                    set_id=set_id,
                    policy="oracle",
                    candidate=c.cell_id,
                    stage=c.stage,
                    count=counts[c.cell_id],
                    total=len(matrix),
                    frequency=counts[c.cell_id] / len(matrix),
                    estimand="canonical_tie_order_oracle_frequency",
                )
            )
    return output, frequencies


def descriptive_tables(data, decisions, headroom):
    grouped = defaultdict(list)
    by_unit = defaultdict(dict)
    for row in decisions:
        grouped[row["set_id"], row["policy"]].append(row)
        grouped["ALL", row["policy"]].append(row)
        by_unit[row["class_id"], row["seed"], row["set_id"]][row["policy"]] = row
    tails, distribution = [], []
    for (set_id, policy), rows in sorted(grouped.items()):
        for metric in ("absolute_regret", "normalized_regret"):
            valid = [r[metric] for r in rows if r[metric] is not None]
            tails.append(
                dict(
                    family="primary",
                    set_id=set_id,
                    policy=policy,
                    metric=metric,
                    units=len(rows),
                    valid_units=len(valid),
                    mean=float(np.mean(valid)) if valid else None,
                    **{
                        name: float(np.quantile(valid, q, method="linear")) if valid else None
                        for name, q in (
                            ("median", 0.5),
                            ("Q90", 0.9),
                            ("Q95", 0.95),
                            ("Q99", 0.99),
                            ("max", 1),
                        )
                    },
                    estimand="raw_unit_tail_descriptive_not_class_bootstrap",
                    unit="class_seed_set",
                )
            )
        if set_id != "ALL":
            members = data["sets"][set_id]
            for c in members:
                # Python 3.12 changed builtin sum(float) accumulation. Use an
                # explicit algorithm for the analytic expected-frequency table.
                count = math.fsum(
                    float(json.loads(r["selected_probability_json"]).get(c.cell_id, 0))
                    for r in rows
                )
                distribution.append(
                    dict(
                        family="primary",
                        set_id=set_id,
                        policy=policy,
                        candidate=c.cell_id,
                        stage=c.stage,
                        count=count,
                        total=len(rows),
                        frequency=count / len(rows),
                        estimand="expected_frequency"
                        if policy == "uniform_random_expected"
                        else "selection_frequency",
                    )
                )
    unit_differences, changes, equivalence = [], [], []
    for left, right in COMPARISONS:
        pair_rows = []
        for (c, s, b), records in sorted(by_unit.items()):
            left_row, right_row = records[left], records[right]
            changed = left_row["selected"] != right_row["selected"]
            row = dict(
                family="primary",
                class_id=c,
                seed=s,
                set_id=b,
                left=left,
                right=right,
                left_selected=left_row["selected"],
                right_selected=right_row["selected"],
                decision_changed=changed,
                absolute_difference=left_row["absolute_regret"] - right_row["absolute_regret"],
                normalized_difference=(
                    left_row["normalized_regret"] - right_row["normalized_regret"]
                    if left_row["resolved"]
                    else None
                ),
                estimand="retrospective_paired_unit_difference",
                unit="class_seed_set",
            )
            pair_rows.append(row)
        unit_differences.extend(pair_rows)
        for b in ["ALL", *sorted(data["sets"])]:
            sub = [r for r in pair_rows if b == "ALL" or r["set_id"] == b]
            changes.append(
                dict(
                    family="primary",
                    set_id=b,
                    left=left,
                    right=right,
                    units=len(sub),
                    changed=sum(r["decision_changed"] for r in sub),
                    helped=sum(r["absolute_difference"] < 0 for r in sub),
                    hurt=sum(r["absolute_difference"] > 0 for r in sub),
                    same_risk=sum(r["absolute_difference"] == 0 for r in sub),
                    estimand="retrospective_choice_change_and_risk_direction",
                )
            )
    for left, right in (
        ("latest_stage", "stage_position_only"),
        ("latest_stage", "pure_G"),
        ("candidate_B", "hybrid"),
        ("candidate_B", "constant"),
    ):
        same = sum(rows[left]["selected"] == rows[right]["selected"] for rows in by_unit.values())
        equivalence.append(
            dict(
                family="primary",
                left=left,
                right=right,
                units=len(by_unit),
                same_selection=same,
                different_selection=len(by_unit) - same,
                exact_selection_alias=same == len(by_unit),
                latest_stage15_sets=sum(
                    max(c.stage for c in members) == 15 for members in data["sets"].values()
                ),
                pure_G_argmin_stage15_sets=sum(
                    data["candidates"][
                        select(
                            members, {c.cell_id: data["frozen"]["G"][c.cell_id] for c in members}
                        )
                    ].stage
                    == 15
                    for members in data["sets"].values()
                ),
                estimand="exact_row_choice_identity_not_mean_equality",
            )
        )
    worst = []
    for policy in POLICIES:
        for i, row in enumerate(
            sorted(
                (r for r in decisions if r["policy"] == policy),
                key=lambda r: (-r["absolute_regret"], r["class_id"], r["seed"], r["set_id"]),
            )[:20],
            1,
        ):
            worst.append(dict(rank=i, **row))
    extremes = []
    for pair in COMPARISONS:
        selected = [r for r in unit_differences if (r["left"], r["right"]) == pair]
        for direction, reverse in (("largest_worsening", True), ("largest_improvement", False)):
            for i, row in enumerate(
                sorted(selected, key=lambda r: r["absolute_difference"], reverse=reverse)[:20], 1
            ):
                extremes.append(dict(rank=i, direction=direction, **row))
    return dict(
        policy_tails=tails,
        policy_selection_frequency=distribution,
        policy_unit_differences=unit_differences,
        decision_changes=changes,
        policy_equivalence=equivalence,
        worst_regret_units=worst,
        largest_policy_differences=extremes,
    )


def _source_reproduction(root, summaries, paired, decisions):
    checks = []
    summary_index = {(r["family"], r["policy"], r["metric"]): r for r in summaries}
    pair_index = {(r["family"], r["left"], r["right"], r["metric"]): r for r in paired}
    for filename, index, keys in (
        ("matched_cost_regret_summary.csv", summary_index, ("family", "predictor", "metric")),
        ("predictor_pairwise_comparison.csv", pair_index, ("family", "left", "right", "metric")),
    ):
        for source in read_csv(root / "evidence/e1pred2r2" / filename):
            if source["family"] != "primary":
                continue
            actual = index[tuple(source[k] for k in keys)]
            for field in ("point", "low", "high", "lower95", "upper95"):
                ref = float(source[field])
                diff = assert_close(actual[field], ref)
                checks.append(
                    dict(
                        source_file=filename,
                        key="/".join(source[k] for k in keys),
                        field=field,
                        main=actual[field],
                        reference=ref,
                        abs_diff=diff,
                        tolerance=1e-12 + 1e-10 * abs(ref),
                        scaled_error=diff / (1e-12 + 1e-10 * abs(ref)),
                    )
                )
    counts = Counter((r["set_id"], r["policy"], r["selected"]) for r in decisions)
    oracle = Counter((r["set_id"], r["oracle"]) for r in decisions if r["policy"] == "constant")
    discrete = 0
    for row in read_csv(root / "evidence/e1pred2r2/decision_diversity.csv"):
        if row["family"] != "primary":
            continue
        count = (
            oracle[row["set_id"], row["cell_id"]]
            if row["predictor"] == "oracle"
            else counts[row["set_id"], row["predictor"], row["cell_id"]]
        )
        if count != int(row["count"]):
            raise AssertionError("Source selection-frequency mismatch")
        discrete += 1
    return checks, discrete


def run_audit(root):
    root = Path(root).resolve()
    data = load_inputs(root)
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, 16, (BOOTSTRAP_DRAWS, 16)
    )
    decisions = build_decisions(data)
    summary, classes, means = summarize_decisions(decisions, indices)
    paired, differences, influence = paired_comparisons(means, indices)
    for row in differences + influence:
        row["class_id"] = data["classes"][row["class_index"]]
    checks, discrete = _source_reproduction(root, summary, paired, decisions)
    headroom, oracle_frequency = headroom_rows(data)
    descriptive = descriptive_tables(data, decisions, headroom)
    subset_sets = [r["set_id"] for r in headroom if r["eligible_exploratory_subset"]]
    subset_rows = [
        dict(r, family="exploratory_oracle_majority_lt_0.75")
        for r in decisions
        if r["set_id"] in subset_sets
    ]
    subset_summary, _, subset_means = (
        summarize_decisions(subset_rows, indices) if subset_rows else ([], [], {})
    )
    subset_paired, _, _ = paired_comparisons(subset_means, indices)
    secondary = []
    for filename, record_type in (
        ("matched_cost_regret_summary.csv", "summary"),
        ("predictor_pairwise_comparison.csv", "paired_comparison"),
    ):
        path = root / "evidence/e1pred2r2" / filename
        for row in read_csv(path):
            if row["family"] != "primary":
                secondary.append(
                    dict(
                        **row,
                        record_type=record_type,
                        analysis_status="source_summary_only",
                        source_path=f"evidence/e1pred2r2/{filename}",
                        source_sha256=file_sha256(path),
                        limitation=(
                            "public membership and frozen Constant cover primary only; "
                            "no new secondary recomputation"
                        ),
                    )
                )
    tables = dict(
        baseline_results=summary,
        decisions=decisions,
        paired_comparisons=paired,
        class_differences=differences,
        class_policy_means=classes,
        leave_one_class_out=influence,
        oracle_headroom=headroom,
        oracle_frequency=oracle_frequency,
        exploratory_subset_results=subset_summary,
        exploratory_subset_comparisons=subset_paired,
        secondary_results=secondary,
        source_reproduction=checks,
        **descriptive,
    )
    tables["selection_costs"] = [
        dict(
            policy=policy,
            incremental_selection_model_calls=0,
            historical_feature_acquisition_model_calls=None,
            required_features=(
                "none"
                if policy in {"latest_stage", "stage_position_only", "uniform_random_expected"}
                else "frozen G only"
                if policy == "pure_G"
                else "frozen source choices"
                if policy == "constant"
                else "source state residual energy"
                if policy == "energy"
                else "source state residual energy and frozen G"
            ),
            cost_scope=(
                "scalar choice with supplied features; "
                "excludes feature acquisition and image generation"
            ),
            historical_cost_status="unknown_in_public_scalar_export; not asserted zero",
            cost_metric="theoretical_weight_saving_bits_not_measured_VRAM_or_latency",
        )
        for policy in POLICIES
    ]
    source_sha = file_sha256(root / "evidence/e1pred2r2/candidate_observations.csv")
    for table in tables.values():
        for row in table:
            row.setdefault("source_observations_sha256", source_sha)
    manifest = dict(
        schema_version=1,
        status="COMPLETED_PUBLIC_SCALAR_AUDIT",
        scope=SCOPE,
        source_base_commit=data["source_base_commit"],
        public_source_commit=data["public_commit"],
        inputs=data["source_inventory"],
        original_public_manifest_sha256=data["public_inventory_sha256"],
        protected_inventory_files_checked=data["protected_inventory_count"],
        classes=data["classes"],
        seeds=data["seeds"],
        candidates=81,
        primary_sets=27,
        decision_units_per_policy=1728,
        policy_rows=len(decisions),
        policies=list(POLICIES),
        beta=data["frozen"]["beta"],
        source_score_checks=5184,
        maximum_source_score_abs_error=data["maximum_source_score_abs_error"],
        source_summary_numeric_checks=len(checks),
        source_diversity_discrete_checks=discrete,
        maximum_source_summary_discrepancy=max(checks, key=lambda r: r["scaled_error"]),
        bootstrap=dict(
            generator="PCG64",
            seed=BOOTSTRAP_SEED,
            draws=BOOTSTRAP_DRAWS,
            shape=list(indices.shape),
            dtype=str(indices.dtype),
            index_matrix_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
            byte_contract="row-major little-endian int64 without header",
            quantile_method="linear",
            two_sided=[0.025, 0.975],
            one_sided=[0.05, 0.95],
            shared_paired_indices=True,
        ),
        tie_rule="lower denoising stage index, source first-seen group order, higher bits, cell ID",
        normalized_range_rule="span > 1e-12*max(max(native_Y),1); undefined otherwise",
        unresolved_decision_rows=sum(not r["resolved"] for r in decisions),
        aggregation="equal cost sets per class/seed; equal four seeds; equal 16 classes",
        headroom_aggregation=(
            "equal class/seed within each cost set; test-best fixed diagnostic only"
        ),
        exploratory_subset=dict(
            selection="test oracle majority < 0.75",
            sets=subset_sets,
            set_count=len(subset_sets),
            units=len(subset_sets) * 64,
            groups=sorted({data["sets"][s][0].group for s in subset_sets}),
            bits=sorted({data["sets"][s][0].bits for s in subset_sets}),
            prospective=False,
        ),
        secondary_status="source_summary_only",
        independent_verification="separate verifier required; not asserted by this main runner",
        scientific_model_calls=0,
        gpu_calls=0,
        fitted_parameters=0,
        unchanged_historical_decisions=True,
        limits=[
            "No raw tensor reconstruction",
            "No model replay or new class/seed observations",
            "No independent leakage or tangent-validity re-audit",
            "No new scientific PASS gate",
            "CI including zero is not equivalence",
            "Oracle entropy is not pairing information",
        ],
    )
    return tables, manifest


def csv_bytes(rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def export_audit(root, destination, *, check=False):
    """Write only the new analysis namespace; check mode does not alter anything."""
    root = Path(root).resolve()
    raw_destination = Path(destination).absolute()
    if any(p.is_symlink() for p in [raw_destination, *raw_destination.parents]):
        raise ValueError("Symlink output locations are not permitted")
    destination = raw_destination.resolve()
    canonical = root / "analysis/baseline_audit"
    if destination == root or root.is_relative_to(destination):
        raise ValueError("Output cannot be the repository or its ancestor")
    if destination.is_relative_to(root) and destination != canonical:
        raise ValueError("In-repository output must use analysis/baseline_audit")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Output must be a directory")
    if destination.exists() and any(path.is_symlink() for path in destination.iterdir()):
        raise ValueError("Symlink output files are not permitted")
    if (
        not check
        and destination != canonical
        and destination.exists()
        and any(destination.iterdir())
    ):
        raise ValueError("External output directory must be new or empty")
    tables, manifest = run_audit(root)
    output_inventory = []
    payloads = {}
    for name, rows in tables.items():
        payload = csv_bytes(rows)
        filename = f"{name}.csv"
        payloads[filename] = payload
        output_inventory.append(
            dict(
                path=filename,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                rows=len(rows),
            )
        )
    manifest["outputs"] = output_inventory
    manifest["implementation"] = [
        dict(
            path=path,
            sha256=file_sha256(root / path),
            bytes=(root / path).stat().st_size,
        )
        for path in ("src/rcfs_dq/baseline_audit.py", "scripts/audit_baselines.py")
    ]
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    files = {**payloads, "audit_manifest.json": manifest_bytes}
    # Validate the whole destination before the first write, including the
    # manifest and symlinks at later positions in the output order.
    for name in files:
        path = destination / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Non-regular output target is not permitted")
    if check:
        for name, payload in files.items():
            if (destination / name).read_bytes() != payload:
                raise AssertionError(f"Recomputed audit output mismatch: {name}")
        return dict(
            passed=True, tables=len(tables), policy_rows=manifest["policy_rows"], model_calls=0
        )
    old_manifest = destination / "audit_manifest.json"
    known = set()
    if destination == canonical and old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if (
            old.get("status") != "COMPLETED_PUBLIC_SCALAR_AUDIT"
            or old.get("scope") != SCOPE
            or old.get("original_public_manifest_sha256")
            != manifest["original_public_manifest_sha256"]
        ):
            raise ValueError("Unrecognized canonical baseline audit")
        for item in old["outputs"]:
            name = item["path"]
            if name not in payloads or name in known or Path(name).name != name:
                raise ValueError("Unsafe or unrecognized previous output inventory")
            path = destination / name
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or file_sha256(path) != item["sha256"]
            ):
                raise ValueError("Modified canonical audit output; preserve and review")
            known.add(name)
    for name, payload in payloads.items():
        path = destination / name
        if path.exists() and name not in known and path.read_bytes() != payload:
            raise ValueError("Unrecognized existing output must not be overwritten")
    if destination == canonical and old_manifest.exists() and not known:
        raise ValueError("Previous canonical output inventory must not be empty")
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (destination / name).write_bytes(payload)
    return dict(
        status=manifest["status"],
        tables=len(tables),
        policy_rows=manifest["policy_rows"],
        model_calls=0,
    )
