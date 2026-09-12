"""Recompute frozen decisions from public scalar records, never model outputs.

Does not update evidence or historical scientific artifacts. Stdout is a new
verification result. Full raw-terminal verification requires excluded tensors.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rcfs_dq.decisions import Candidate, class_block_ci, evaluate_regret, exact_cost_sets, select
from rcfs_dq.geometry import log_contrasts
from rcfs_dq.scoring import FrozenCandidateB
from rcfs_dq.verification import assert_close, independent_regret, verify_inventory

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify():
    discrepancies = []

    def check(actual, reference, key):
        error = assert_close(actual, reference)
        tolerance = 1e-12 + 1e-10 * abs(reference)
        discrepancies.append(
            dict(
                key=key,
                actual=float(actual),
                reference=float(reference),
                absolute_error=error,
                tolerance=tolerance,
                scaled_error=error / tolerance,
            )
        )

    manifest = json.loads((ROOT / "PUBLIC_EVIDENCE_MANIFEST.json").read_text())
    inventory = verify_inventory(ROOT, manifest)
    frozen = json.loads((ROOT / "configs/frozen_predictor.json").read_text())
    score = FrozenCandidateB(frozen["G"], frozen["beta"])
    candidates = {}
    group_order = {}
    for r in rows("configs/candidates.csv"):
        group_order.setdefault(r["group"], len(group_order))
        c = Candidate(
            r["cell_id"],
            r["group"],
            int(r["stage"]),
            int(r["bits"]),
            int(r["numel"]),
            group_order[r["group"]],
            r["regime"],
        )
        assert c.packed_bits == int(r["packed_bits"]) and c.saving_bits == int(r["saving_bits"])
        shapes = json.loads(r["shapes"])
        assert sum(int(np.prod(s)) for s in shapes.values()) == c.numel
        assert c.cell_id not in candidates
        candidates[c.cell_id] = c
    sets = defaultdict(list)
    for r in rows("configs/cost_sets.csv"):
        sets[r["set_id"]].append(candidates[r["cell_id"]])
    assert len(sets) == len(exact_cost_sets(list(candidates.values()))) == 27
    assert {tuple(sorted(c.cell_id for c in v)) for v in sets.values()} == {
        tuple(sorted(c.cell_id for c in v))
        for v in exact_cost_sets(list(candidates.values())).values()
    }
    observations = defaultdict(dict)
    for r in rows("evidence/e1pred2r2/candidate_observations.csv"):
        key = int(r["class_id"]), int(r["seed"])
        assert r["cell_id"] not in observations[key]
        observations[key][r["cell_id"]] = r
        check(
            score.score(r["cell_id"], float(r["energy"])),
            float(r["source_B_score"]),
            f"score/{key}/{r['cell_id']}",
        )
    assert len(observations) == 64 and all(set(v) == set(candidates) for v in observations.values())
    by_class = defaultdict(list)
    for (class_id, seed), records in sorted(observations.items()):
        for set_id, members in sorted(sets.items()):
            e = {c.cell_id: float(records[c.cell_id]["energy"]) for c in members}
            b = {c.cell_id: score.score(c.cell_id, e[c.cell_id]) for c in members}
            y = {c.cell_id: float(records[c.cell_id]["native_Y"]) for c in members}
            hybrid_scores = {
                c.cell_id: e[c.cell_id] if c.regime == "M3" else b[c.cell_id] for c in members
            }
            selections = dict(
                constant=frozen["constants"][set_id],
                energy=select(members, e),
                candidate_B=select(members, b),
                hybrid=select(members, hybrid_scores, hybrid=True),
            )
            for policy, choice in selections.items():
                result = evaluate_regret(members, y, choice)
                reference = independent_regret(
                    y, choice, [c.cell_id for c in sorted(members, key=lambda c: c.tie_key)]
                )
                assert result["resolved"] and result["oracle"] == reference["oracle"]
                for metric in ["absolute_regret", "normalized_regret"]:
                    check(
                        result[metric],
                        reference[metric],
                        f"independent/{class_id}/{seed}/{set_id}/{policy}/{metric}",
                    )
                    by_class[policy, metric, class_id].append((seed, set_id, result[metric]))
                by_class[policy, "oracle_hit", class_id].append(
                    (seed, set_id, float(choice == result["oracle"]))
                )
    classes = sorted({c for c, _ in observations})
    assert len(classes) == 16
    means = {}
    for policy in selections:
        for metric in ["absolute_regret", "normalized_regret", "oracle_hit"]:
            values = []
            for c in classes:
                units = by_class[policy, metric, c]
                assert len(units) == 108
                per_seed = defaultdict(list)
                for seed, _, value in units:
                    per_seed[seed].append(value)
                assert len(per_seed) == 4 and all(len(v) == 27 for v in per_seed.values())
                values.append(np.mean([np.mean(v) for v in per_seed.values()]))
            means[policy, metric] = np.asarray(values)
    indices = np.random.Generator(np.random.PCG64(2026091101)).integers(0, 16, (10000, 16))
    for r in rows("evidence/e1pred2r2/matched_cost_regret_summary.csv"):
        if r["family"] != "primary":
            continue
        computed = class_block_ci(means[r["predictor"], r["metric"]], indices=indices)
        for field, value in computed.items():
            check(value, float(r[field]), f"summary/{r['predictor']}/{r['metric']}/{field}")
    for r in rows("evidence/e1pred2r2/predictor_pairwise_comparison.csv"):
        if r["family"] != "primary":
            continue
        delta = means[r["left"], r["metric"]] - means[r["right"], r["metric"]]
        for field, value in class_block_ci(delta, indices=indices).items():
            check(value, float(r[field]), f"paired/{r['left']}/{r['right']}/{r['metric']}/{field}")
    discovery = [
        r
        for r in rows("evidence/e1d1/directional_contrasts.csv")
        if r["alpha_role"] == "fixed_alpha"
    ]
    assert len(discovery) == 81
    assert all(r["cell_tangent_status"] == "DISCOVERY_TANGENT_SUPPORTED" for r in discovery)
    for r in discovery:
        v = log_contrasts(
            float(r["LCG_actual"]),
            float(r["LCG_shuffled"]),
            [float(r["LCG_iso0"]), float(r["LCG_iso1"])],
        )
        check(v["L_EDG_state"], float(r["L-EDG-state"]), f"discovery/{r['cell_id']}")
    positive = next(
        r for r in rows("evidence/e1c/global_tests.csv") if r["test"] == "positive_sentinel_global"
    )
    cells = set(positive["cells"].split(","))
    confirmation = [
        r for r in rows("evidence/e1c/directional_replication.csv") if r["cell_id"] in cells
    ]
    assert len(confirmation) == len(cells) == 9
    assert sum(r["same_sign"] == "True" for r in confirmation) == 9
    assert sum(r["replicated"] == "True" for r in confirmation) == 8
    check(
        np.mean([float(r["confirmation_class_mean"]) for r in confirmation]),
        float(positive["class_effect_mean"]),
        "confirmation/positive_family_mean",
    )
    tail = json.loads((ROOT / "evidence/e1tail0/analysis_summary.json").read_text())
    d = tail["detection"]
    assert tail["provisional_decision"] == "E1TAIL0_NO_USEFUL_EXCEPTION_SIGNAL"
    assert (d["TP"], d["FP"], d["FN"], d["TN"]) == (3, 474, 76, 1175)
    check(d["recall"], 3 / 79, "tail/recall")
    check(d["FPR"], 474 / 1649, "tail/FPR")
    return dict(
        passed=True,
        inventory_files=inventory,
        scalar_comparisons=len(discrepancies),
        maximum_scaled_discrepancy=max(discrepancies, key=lambda r: r["scaled_error"]),
        maximum_absolute_discrepancy=max(discrepancies, key=lambda r: r["absolute_error"]),
        confirmation_signs=9,
        confirmation_stricter_replications=8,
        maps=64,
        primary_sets=27,
        decision_units=1728,
        policies=4,
        mean_regret={
            p: {m: float(means[p, m].mean()) for m in ["absolute_regret", "normalized_regret"]}
            for p in selections
        },
        scope="public scalar recomputation, not raw-terminal or full experiment replay",
        model_calls=0,
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
