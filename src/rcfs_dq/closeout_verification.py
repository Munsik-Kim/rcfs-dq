"""Independent scalar reference for the published V3/BRIDGE0 closeout layer.

No research runner, main prediction function, model, Torch or private raw IO is
imported. Stored norms/errors are inputs, not independently regenerated vectors.
"""

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

CLASSES = (123, 206, 325, 408, 569, 701, 789, 887)
SEEDS = (107, 108, 109, 110)
GROUPS = ("qkv_all_blocks", "attention_output_all_blocks", "mlp_down_all_blocks")
FAMILIES = ("actual_local", "accumulated", "donor", "isotropic")
EVENTS = (4, 10, 15)
ALPHAS = (1 / 128, 1 / 64, 1 / 32)


def require(condition, label):
    if not condition:
        raise ValueError(label)


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    require(np.isfinite(result), "Nonfinite public scalar")
    return result


def flag(value):
    require(value in ("True", "False"), "Boolean schema")
    return value == "True"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def vkey(row):
    return (
        int(row["class_id"]),
        int(row["seed"]),
        row["group"],
        int(row["event_stage"]),
        row["family"],
        float(row["alpha"]),
    )


def bkey(row):
    return (
        int(row["class_id"]),
        int(row["seed"]),
        row["group"],
        int(row["origin_event"]),
        int(row["probe_boundary"]),
    )


class Checks:
    def __init__(self):
        self.count = 0
        self.maximum = {"scaled_error": 0.0, "absolute_error": 0.0, "key": "exact"}

    def equal(self, actual, expected, key):
        self.count += 1
        if actual is None or expected is None:
            require(actual is expected, "Undefined scalar mismatch: " + key)
            return
        error = abs(float(actual) - float(expected))
        tolerance = 1e-12 + 1e-10 * abs(float(expected))
        scaled = error / tolerance
        if scaled > self.maximum["scaled_error"]:
            self.maximum = {"key": key, "absolute_error": error, "scaled_error": scaled}
        require(error <= tolerance, "Scalar mismatch: " + key)


def average(values, median=False):
    if not values or any(v is None for v in values):
        return None
    return float(np.median(values) if median else np.mean(values))


def interval(values, indices, quantiles):
    if any(v is None for v in values):
        return [None] * (1 + len(quantiles))
    values = np.asarray(values, dtype=np.float64)
    sampled = np.mean(values[indices], axis=1)
    return [float(values.mean()), *map(float, np.quantile(sampled, quantiles, method="linear"))]


def manifest_check(root):
    manifest = read(root / "PUBLIC_CLOSEOUT_MANIFEST.json")
    names = set()
    for item in manifest["files"]:
        name = item["path"]
        path = root / name
        require(name not in names, "Duplicate manifest member")
        names.add(name)
        require(
            not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Unsafe path"
        )
        data = path.read_bytes()
        require(
            len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
            "Closeout payload hash mismatch",
        )
    layer = root / "analysis/closeout"
    for item in read(layer / "SOURCE_PROVENANCE.json")["copied_files"]:
        require(
            hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"],
            "Source-copy certificate mismatch",
        )
    preservation = read(layer / "PRESERVATION_MANIFEST.json")
    for item in preservation["protected_files"]:
        data = (root / item["path"]).read_bytes()
        require(
            len(data) == item["bytes"]
            and hashlib.sha256(data).hexdigest() == item["sha256"] == item["after_sha256"],
            "Original payload changed",
        )
    return len(names), len(preservation["protected_files"])


def verify_v3(root, checks):
    folder = root / "analysis/closeout/v3"
    finite, predictions, qualification = [
        rows(folder / (n + ".csv"))
        for n in ("finite_difference_rows", "prediction_rows", "qualification_rows")
    ]
    expected = set(product(CLASSES, SEEDS, GROUPS, EVENTS, FAMILIES, ALPHAS))
    for table in (finite, predictions, qualification):
        require(len(table) == len(expected) and {vkey(r) for r in table} == expected, "V3 coverage")
    centers = {vkey(r)[:-1]: r for r in finite if float(r["alpha"]) == 1 / 64}
    targets = {vkey(r)[:-1]: r for r in predictions if float(r["alpha"]) == 1 / 128}
    cfail, dfail = 0, 0
    c_families = dict.fromkeys(FAMILIES, True)
    for r in qualification:
        c = number(r["relative_derivative_drift"]) <= 0.15 and number(r["direction_cosine"]) >= 0.99
        d = number(r["odd_curvature"]) <= 0.15
        require(
            c == flag(r["C_pass"]) and d == flag(r["D_source_ratio_within_reference"]), "V3 C/D"
        )
        if not (r["event_stage"] == "4" and r["family"] == "accumulated"):
            cfail += not c
            dfail += not d
        c_families[r["family"]] &= c
    decision = read(folder / "DECISION.json")
    require(
        c_families == decision["C_by_family"]
        and dfail == decision["D_source_ratio_failures_unique_alpha_rows"],
        "V3 qualifications",
    )
    unique_finite = [
        r for r in finite if not (r["event_stage"] == "4" and r["family"] == "accumulated")
    ]
    require(len(unique_finite) == decision["unique_alpha_rows"], "V3 aliases")
    for field, recorded in (
        ("R_pass", "R_failed_unique_alpha_rows"),
        ("N_pass", "N_failed_unique_alpha_rows"),
    ):
        require(
            sum(not flag(r[field]) for r in unique_finite) == decision[recorded],
            "V3 R/N recorded flags",
        )
    for r in predictions:
        for policy in ("noJ", "scalar", "full"):
            norm = number(r["target_norm"])
            expected_ratio = number(r["err_" + policy]) / norm if norm > number(r["tau"]) else None
            checks.equal(
                number(r["normalized_error_" + policy]), expected_ratio, "V3 normalized error"
            )
    metrics = (
        "R_J",
        "err_noJ",
        "err_scalar",
        "err_full",
        "normalized_error_noJ",
        "normalized_error_scalar",
        "normalized_error_full",
        "relative_improvement_full_vs_noJ",
    )

    def class_value(cls, value):
        return average(
            [
                average([value((cls, s, g, e)) for s in SEEDS], median=True)
                for g in GROUPS
                for e in EVENTS
            ]
        )

    def metric_at(key, family, metric):
        return number((centers if metric == "R_J" else targets)[(*key, family)][metric])

    endpoints = {
        (r["family"], int(r["class_id"]), r["metric"]): number(r["value"])
        for r in rows(folder / "class_endpoints.csv")
    }
    derived = {}
    for fam, metric in product(FAMILIES, metrics):
        values = [class_value(c, lambda k: metric_at(k, fam, metric)) for c in CLASSES]
        derived[fam, metric] = values
        for c, v in zip(CLASSES, values, strict=True):
            checks.equal(endpoints[fam, c, metric], v, f"V3 class/{fam}/{c}/{metric}")
    indices = np.random.Generator(np.random.PCG64(2026091501)).integers(0, 8, size=(10000, 8))
    names = ("point", "lower95", "upper95", "lower_two95", "upper_two95")
    for r in rows(folder / "bootstrap_summary.csv"):
        expected_ci = interval(
            derived[r["family"], r["metric"]], indices, [0.05, 0.95, 0.025, 0.975]
        )
        for n, v in zip(names, expected_ci, strict=True):
            checks.equal(number(r[n]), v, "V3 summary/" + r["family"] + "/" + r["metric"] + "/" + n)
    contrasts = {}
    for r in rows(folder / "paired_contrasts.csv"):

        def difference(k):
            if r["metric"] == "R_J":
                return metric_at(k, r["left"], "R_J") - metric_at(k, r["right"], "R_J")
            return metric_at(k, "actual_local", "err_" + r["left"]) - metric_at(
                k, "actual_local", "err_" + r["right"]
            )

        expected_ci = interval(
            [class_value(c, difference) for c in CLASSES],
            indices,
            [0.05, 0.95, 0.025, 0.975, 0.005, 0.995],
        )
        for n, v in zip(
            (*names, "lower_familywise95", "upper_familywise95"), expected_ci, strict=True
        ):
            checks.equal(number(r[n]), v, "V3 contrast/" + r["contrast"] + "/" + n)
        contrasts[r["contrast"]] = {"point": expected_ci[0], "two_sided95": expected_ci[3:5]}
    for r in rows(folder / "prediction_tail_summary.csv"):
        prefix = "err_" if r["metric"] == "absolute_error" else "normalized_error_"
        values = [
            number(z[prefix + r["predictor"]])
            for z in predictions
            if z["family"] == r["family"]
            and z["alpha"] == r["alpha"]
            and not (z["event_stage"] == "4" and z["family"] == "accumulated")
        ]
        require(len(values) == int(r["rows"]), "V3 tail denominator")
        expected_tail = [
            np.mean(values),
            *np.quantile(values, [0.5, 0.9, 0.95, 0.99, 1], method="linear"),
        ]
        for name, value in zip(
            ("mean", "median", "Q90", "Q95", "Q99", "max"), expected_tail, strict=True
        ):
            checks.equal(number(r[name]), value, "V3 tail/" + name)
    return {
        "displayed_rows": len(finite),
        "unique_rows": len(unique_finite),
        "C_failures": cfail,
        "D_failures": dfail,
        "C_by_family": c_families,
        "contrasts": contrasts,
        "R_J_actual_local": float(np.mean(derived["actual_local", "R_J"])),
    }


def verify_bridge(root, checks):
    folder = root / "analysis/closeout/bridge0"
    source = rows(folder / "natural_error_rows.csv")
    expected = set(
        product(CLASSES, SEEDS, GROUPS, (4,), (5, 11, 16), ("operational", "scheduler64"))
    )
    require(
        len(source) == len(expected) and {(*bkey(r), r["target_kind"]) for r in source} == expected,
        "Bridge intended target coverage",
    )
    available = [r for r in source if r["status"] == "ANALYZED"]
    require(len(available) == 384, "Bridge available coverage")
    for r in source:
        require(
            (r["status"] == "NOT_STORED") == (r["probe_boundary"] == "16"), "Missing target binding"
        )
        require(
            int(r["source_event_label"]) == int(r["probe_boundary"]) - 1, "Origin/probe bookkeeping"
        )
        if r["status"] == "NOT_STORED":
            require(r["target_norm"] == "" and r["err_abs_full"] == "", "Missing target not null")
            continue
        for p in ("noJ", "scalar", "full"):
            err = number(r["err_abs_" + p])
            checks.equal(number(r["err_sq_" + p]), err**2, "Bridge square")
            denominator = number(r["target_norm"])
            rel = err / denominator if denominator > number(r["tau"]) else None
            checks.equal(number(r["err_rel_" + p]), rel, "Bridge relative")
        for left, right in (("full", "scalar"), ("full", "noJ"), ("scalar", "noJ")):
            checks.equal(
                number(r[f"abs_{left}_minus_{right}"]),
                number(r["err_abs_" + left]) - number(r["err_abs_" + right]),
                "Bridge paired",
            )
    indices = np.random.Generator(np.random.PCG64(2026091502)).integers(0, 8, size=(10000, 8))
    indexed = {(*bkey(r), r["target_kind"]): r for r in available}

    def scope_specs(scope):
        scope = scope.removeprefix("breakdown:")
        return [
            (g, p)
            for g in GROUPS
            for p in (5, 11)
            if scope == "available_balanced" or scope == f"probe{p}" or scope == f"{g}:probe{p}"
        ]

    def values(metric, target, estimator, scope="available_balanced"):
        specs = scope_specs(scope)
        require(bool(specs), "Unknown bridge scope")
        return [
            average(
                [
                    average(
                        [number(indexed[c, s, g, 4, p, target][metric]) for s in SEEDS],
                        median=estimator == "median",
                    )
                    for g, p in specs
                ]
            )
            for c in CLASSES
        ]

    for r in rows(folder / "class_endpoints.csv"):
        v = values(r["metric"], r["target_kind"], r["estimator"])
        checks.equal(number(r["value"]), v[CLASSES.index(int(r["class_id"]))], "Bridge class")
    primary = None
    for r in rows(folder / "paired_comparisons.csv"):
        vals = values(r["metric"], r["target_kind"], r["estimator"], r["scope"])
        expected_ci = interval(vals, indices, [0.025, 0.975])
        for name, value in zip(("point", "lower95", "upper95"), expected_ci, strict=True):
            checks.equal(number(r[name]), value, "Bridge interval/" + r["metric"] + "/" + name)
        if (r["scope"], r["metric"], r["target_kind"], r["estimator"]) == (
            "available_balanced",
            "abs_full_minus_scalar",
            "operational",
            "mean",
        ):
            primary = expected_ci
    for r in rows(folder / "tails_and_influence.csv"):
        if r["kind"] == "class_influence":
            vals = values("abs_full_minus_scalar", r["target_kind"], "mean")
            v = [x for c, x in zip(CLASSES, vals, strict=True) if c != int(r["excluded_class"])]
            checks.equal(number(r["leave_one_class_out_point"]), average(v), "Bridge influence")
        else:
            specs = scope_specs(r["scope"])
            vals = [
                number(z["err_abs_" + r["policy"]])
                for z in available
                if z["target_kind"] == r["target_kind"]
                and (z["group"], int(z["probe_boundary"])) in specs
            ]
            require(len(vals) == int(r["rows"]), "Bridge tail count")
            for name, value in zip(
                ("mean", "Q90", "Q95", "Q99", "max"),
                [np.mean(vals), *np.quantile(vals, [0.9, 0.95, 0.99, 1], method="linear")],
                strict=True,
            ):
                checks.equal(number(r[name]), value, "Bridge tail/" + name)
    op = [r for r in available if r["target_kind"] == "operational"]
    signs = [
        sum(bool(np.sign(number(r["abs_full_minus_scalar"])) == s) for r in op) for s in (-1, 1, 0)
    ]
    decision = read(folder / "DECISION.json")
    require(
        decision["three_probe_summary"] is None and decision["available_targets"] == len(op),
        "Bridge missing scope decision",
    )
    checks.equal(decision["primary_point"], primary[0], "Bridge decision point")
    for a, b in zip(decision["primary_CI95"], primary[1:], strict=True):
        checks.equal(a, b, "Bridge decision CI")
    digest = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    require(digest == decision["bootstrap_index_sha256"], "Bridge bootstrap identity")
    return {
        "available": len(op),
        "missing": 96,
        "three_probe_summary": None,
        "Full_minus_Scalar": primary,
        "help_hurt_tie": signs,
        "bootstrap_index_sha256": digest,
    }


def verify_closeout(root):
    root = Path(root)
    files, protected = manifest_check(root)
    checks = Checks()
    v3 = verify_v3(root, checks)
    bridge = verify_bridge(root, checks)
    status = read(root / "analysis/closeout/PROJECT_STATUS.json")
    require(
        status["project_level_closeout"] == "RCFS_DQ_LOCAL_MECHANISM_TRACK_CLOSED_AS_LIMITED",
        "Closeout scope",
    )
    require(not status["new_scientific_execution_authorized"], "No science authorization")
    old = root / "analysis/closeout/historical"
    require(read(old / "J0_V1_OUTCOME.json")["failed_radius_rows"] == 157, "v1 preserved failures")
    require(
        read(old / "NR0_DECISION.json")["decision"]
        == status["historical_scientific_statuses"]["NR0"],
        "NR0 preserved",
    )
    return {
        "passed": True,
        "scope": "independent public scalar arithmetic; not raw-vector/model reproduction",
        "files_checked": files,
        "protected_files_checked": protected,
        "scalar_comparisons": checks.count,
        "maximum_discrepancy": checks.maximum,
        "V3": v3,
        "BRIDGE0": bridge,
        "historical_raw_verification_receipts": "source records only, not rerun by this verifier",
        "model_calls": 0,
        "GPU_calls": 0,
        "predictor_refits": 0,
    }
