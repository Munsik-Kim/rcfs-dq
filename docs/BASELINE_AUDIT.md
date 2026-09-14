# P0: free and static decision baselines

This is a retrospective CPU-only reanalysis of existing public scalar evidence.
Model calls = 0; GPU calls = 0; new model observations = 0; predictor refits = 0;
new prospective confirmation = 0. Original scientific values, policies and
decisions remain unchanged. Neither the main calculation nor the independent
NumPy verifier regenerates raw terminal tensors or reestablishes leakage and
tangent validity from scratch.
The original prospective primary endpoint and recorded decision are unchanged.
Added static comparators and absolute-risk diagnostics are retrospective
limitations, not a reclassification of the completed prospective experiment.

## Inputs and separate provenance

The public source anchor is `af7b402839d1052ece994c80a91e59e10d7474dc`.
Original observations, diversity, regret summaries and pairwise intervals come
from `evidence/e1pred2r2/`; candidate membership and frozen scores come from
`configs/`. These are protected by the unchanged
[original evidence manifest](../PUBLIC_EVIDENCE_MANIFEST.json).
The separate [audit manifest](../PUBLIC_AUDIT_MANIFEST.json) inventories derived
outputs, source hashes, implementation and scope. Per-analysis inputs and
estimands are in [audit_manifest.json](../analysis/baseline_audit/audit_manifest.json).

The observation SHA-256 is
`9ed386af3e565cf6ed8b4cf6119bb9126795d5b67de8900f021e06860f45d4dd`;
the frozen predictor SHA-256 is
`7c7d42eb59eb088a5e96bc4336f5614aba2097e14c04ec9e17e11d81d580cd7e`.
The complete grid has 81 candidates, 16 classes, four seeds, 27 three-candidate
exact-cost sets and 1,728 decision units. Missing, duplicate, nonfinite,
negative-risk or mismatched-cost inputs are rejected, not dropped or replaced.

## Policy definitions and cost

| Policy | Information used for safe-candidate selection |
|---|---|
| Source Constant | Development-only frozen choice per set, independent of test class/seed |
| Energy | Minimum source-defined local quantization-response energy E |
| Candidate-B | Minimum `E * G**beta`, using frozen G and beta 0.6735229330314306 |
| Hybrid | Frozen M3 high-risk guard; Energy within M3, Candidate-B in L78 |
| Latest-stage | Largest denoising index, not largest reverse scheduler timestep |
| Stage-position-only | Minimum `-stage_index`; an explicit alias of latest-stage |
| Pure-G | Minimum frozen G, without test E |
| Uniform-random expectation | Equal probability across three candidates, evaluated analytically |

Score ties follow `(stage, group_order, -bits, cell_id)` after any frozen regime
guard. Selection does not receive terminal Y. Oracle choice and post-selection
regret use Y. Uniform expectation stores no sampled selected candidate; canonical
oracle-hit expectation is 1/3, not a post-hoc tie-expanded success rule.

Cost is exact `numel * (16 - bits)` theoretical FP16-relative weight saving.
It does not mean equal measured latency, compressed resident memory or packed
kernel speed. Latest-stage and source Constant require zero additional online
selection forwards, but generating an image is not free. Pure-G and Constant
still have historical development costs; Energy/B require residual acquisition.
No unledgered acquisition budget is assumed to be zero. See
[selection_costs.csv](../analysis/baseline_audit/selection_costs.csv).

## Regret, aggregation and uncertainty

Absolute regret is selected Y minus the tied oracle minimum. For
`span = max(Y) - min(Y)` and `tau = 1e-12 * max(max(Y), 1)`, normalized regret is
`regret / span` only when `span > tau`; otherwise it is undefined and absolute
regret is retained. This algebraic resolution floor is not measured numerical
noise. All 1,728 current units are resolved.

The estimator is an equal 27-set mean within class/seed, then an equal four-seed
mean, then an equal 16-class mean. Raw Q90/Q95/Q99/max are descriptive tails.
The source bootstrap convention is PCG64(2026091101), 10,000 shared 16-class
resamples. The index matrix SHA-256 is
`2f110408df13144fbe6302ff3d52705bcb1965734593261b54c7e78c3e760a77`.
Linear 0.025/0.975 quantiles give two-sided 95% intervals; separately stored
0.05/0.95 quantiles give one-sided bounds. The 1,728 units are not independent
population samples. Numeric comparison uses
`abs(main-ref) <= 1e-12 + 1e-10 * abs(ref)`; keys and choices agree exactly.

## Findings and conditional headroom

[RESULTS](RESULTS.md) gives mean absolute/normalized regret, paired intervals,
oracle-hit rates and tails. Candidate-B improves over Energy. Its normalized
intervals versus Constant and latest-stage cross zero, whereas B−latest absolute
regret is +0.166608 [+0.024959, +0.388829]. A lower normalized point does not
establish lower absolute risk, and crossing zero is not proof of equivalence.

Pure-G, latest-stage and negative-stage-position choices match in all 1,728
units; G is minimized at stage15 in all 27 sets. B and Hybrid also coincide.
These aliases concern actual selections, not merely rounded mean equality.
They do not imply that relative G values cannot change `E * G**beta` choices;
[P1](PAIRING_ABLATION.md) tests a stage-coefficient alternative explicitly.

Stage15 is the modal oracle in 25/27 sets. Fourteen sets have an entirely fixed
oracle in all 64 class/seed units, with zero empirical oracle-switching headroom
up to arithmetic cancellation tolerance. The majority-rate distribution across
sets is: 0.46875, 0.5, 0.515625, 0.578125, 0.734375 and 0.8125 once each;
0.953125 three times; 0.984375 four times; and 1.0 fourteen times.
Thus 22/27 sets have majority rate at least 0.75, a descriptive concentration
count, not a new low-risk or PASS threshold.

The empirical best fixed candidate for mean absolute risk is stage15 in all
27 sets. Its equal-set mean headroom is 0.019874962; the separately optimized
normalized headroom is 0.045282708. These test-best-fixed policies are not
offered as deployment baselines. Modal oracle and minimum expected risk differ
in two sets, so oracle frequency alone is insufficient.

The five outcome-defined `majority < 0.75` sets (320 units) are attention-output
W4/W6, MLP-up W4/W6 and adaLN-conditioning W6. Their analysis is exploratory;
it never replaces the full-grid denominator. All worst units, harmful/helpful
changes and leave-one-class-out diagnostics remain available without removing
classes. In particular, B's maximum absolute regret is 186.507995 versus
latest-stage's 2.970177; its lower Q95 but higher Q99/max is not labeled tail safety.

The separate source secondary comparison is unfavorable to Hybrid:
Hybrid−Constant normalized regret +0.0007820371
[+0.0000246697, +0.0017184796]. Secondary membership/Constant data are incomplete
in the public package, so this is `source_summary_only` readback, not a new
reconstruction or part of the primary mean.

## Reproduce and inspect

With the repository's pinned dependencies, these checks do not change payloads:

```bash
python scripts/verify_evidence.py
python -O scripts/verify_evidence.py
python scripts/audit_baselines.py --check
python scripts/verify_baseline_audit.py --check
python scripts/verify_public_audits.py
```

The main audit can write a fresh export via `--output` to a new or empty directory.
Do not use original `evidence/`, `configs/` or `figures/` as output destinations.
The public verifier independently derives selections and statistics from scalar
inputs rather than reading document numbers or copying main summary flags.

- [Policy results](../analysis/baseline_audit/baseline_results.csv),
  [paired contrasts](../analysis/baseline_audit/paired_comparisons.csv),
  [class differences](../analysis/baseline_audit/class_differences.csv)
- [Aliases](../analysis/baseline_audit/policy_equivalence.csv),
  [headroom](../analysis/baseline_audit/oracle_headroom.csv),
  [oracle frequency](../analysis/baseline_audit/oracle_frequency.csv)
- [Tails](../analysis/baseline_audit/policy_tails.csv),
  [largest differences](../analysis/baseline_audit/largest_policy_differences.csv),
  [class influence](../analysis/baseline_audit/leave_one_class_out.csv)
- [Exploratory subset](../analysis/baseline_audit/exploratory_subset_results.csv),
  [secondary source](../analysis/baseline_audit/secondary_results.csv),
  [independent verification](../analysis/baseline_audit/independent_verification.json)

These are reproducibility checks, not requirements that Candidate-B must win.
