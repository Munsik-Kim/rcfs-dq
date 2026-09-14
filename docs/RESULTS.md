# Results and claim boundaries

These are compact records of completed source experiments, not new model
observations generated during public packaging. All included original tables
and figures retain their source bytes; added scalar/config projections retain
their source values. See the manifest and reproducibility scope.

## Residual-conditioned geometry

Core81 discovery met the locked tangent-support criterion in 81/81 cells.
Independent confirmation's preselected positive family had mean state contrast
1.9091539, exact one-sided p = 0.00390625, and bootstrap interval
[1.74956, 2.06824]. Expected signs held in 9/9 cells; 8/9 met the stricter
replication criteria. “Stricter” refers to the statistical criteria, not a larger
effect than discovery. Evidence is conditional on the selected cells and panels.

## Original prospective native-FP16 matched-cost decisions

The fresh 16-class × 4-seed panel contains 1,728 class/seed/set decision units
across 27 same-group/same-bit exact-cost stage-selection sets. Uncertainty uses
16 class blocks, not 1,728 independent population samples.

The original Energy, source Constant, Candidate-B and Hybrid policies were frozen
before these observations. Their source values and intervals remain unchanged;
the audit below recomputes them alongside retrospective comparison policies.
The original prospective primary endpoint and recorded scientific decision are
not redefined: static-baseline and absolute-risk diagnostics add retrospective
limitations, not replacement success criteria for the completed experiment.

These are theoretical packed-storage-matched, single-intervention decisions.
They are not measured memory/latency savings or a full allocation result.

## Retrospective decision audit: trivial baselines and pairing ablation

This section is a CPU-only reanalysis of already-published scalar observations.
It contains no new model execution and is not a prospective confirmation experiment.
P0/P1 model calls = 0, GPU calls = 0, new model observations = 0,
predictor refits = 0 and new prospective confirmation = 0.

Original payloads remain under
[PUBLIC_EVIDENCE_MANIFEST.json](../PUBLIC_EVIDENCE_MANIFEST.json). The separate
[PUBLIC_AUDIT_MANIFEST.json](../PUBLIC_AUDIT_MANIFEST.json) inventories derived
results, scripts, source inputs and the P1 historical scalar provenance certificate.
Neither layer claims to regenerate excluded raw model outputs.

### P0: static and trivial policies

| Policy | Mean absolute regret | Mean normalized regret |
|---|---:|---:|
| Energy | 0.292808 | 0.185153 |
| Source Constant | 0.210804 | 0.046017 |
| Latest-stage | 0.019875 | 0.052227 |
| Pure-G | 0.019875 | 0.052227 |
| Candidate-B | 0.186483 | 0.043135 |
| Hybrid | 0.186483 | 0.043135 |
| Uniform-random expectation | 25.991948 | 0.424144 |

Latest-stage chooses the largest denoising index, not scheduler timestep.
Pure-G minimizes the original frozen G without test residual energy. Its choices
and the negative-stage-position rule match latest-stage on all 1,728 units.
Hybrid and Candidate-B also coincide on every primary unit. Uniform random is
an analytic expectation over each three-candidate set, not a favorable random draw.

Each mean averages 27 cost sets within class/seed, four seeds within class, and
16 classes equally. All 1,728 units have resolved risk ranges. Paired intervals
use PCG64(2026091101), 10,000 shared 16-class resamples and linear 0.025/0.975
quantiles. Negative differences mean lower regret; these are conditional
retrospective intervals, not new confirmatory gates.

| Comparison | Absolute difference [95% CI] | Normalized difference [95% CI] |
|---|---:|---:|
| Candidate-B − Energy | −0.106325 [−0.167860, −0.057550] | −0.142018 [−0.152157, −0.131562] |
| Candidate-B − Constant | −0.024321 [−0.048095, −0.005405] | −0.002882 [−0.007260, +0.001869] |
| Candidate-B − Latest-stage | +0.166608 [+0.024959, +0.388829] | −0.009091 [−0.020609, +0.002968] |

The substantial improvement over Energy survives. Strict normalized-regret
superiority over Constant or latest-stage is not established. B−Constant has
a favorable absolute-regret interval, but that is not evidence that the policy
beats all static baselines: latest-stage has much lower mean absolute regret.
B−latest worsens absolute regret in 12/16 classes. The normalization weights
risk ranges differently; a favorable normalized point estimate cannot stand
in for lower absolute loss. Interval overlap with zero is not proof of equivalence.

Raw-unit absolute-risk quantiles below are descriptive, not 1,728-independent-unit
confidence intervals. Hybrid shares Candidate-B's row; Pure-G and stage-position
share latest-stage's row. Full normalized tails and per-set records are also
preserved in [policy_tails.csv](../analysis/baseline_audit/policy_tails.csv).

| Policy | Oracle hit rate | Q90 absolute | Q95 absolute | Q99 absolute | Maximum absolute |
|---|---:|---:|---:|---:|---:|
| Energy | 0.632523 | 0.109912 | 0.487306 | 3.312601 | 186.507995 |
| Source Constant | 0.902199 | 0.000000 | 0.038122 | 2.058133 | 186.507995 |
| Latest-stage | 0.890625 | 0.003058 | 0.043997 | 0.597707 | 2.970177 |
| Candidate-B | 0.906250 | 0.000000 | 0.025274 | 1.527430 | 186.507995 |

Candidate-B's lower Q95 but higher Q99/max than latest-stage is retained.
The largest B−latest worsening is the historical class85/seed82 MLP-up W4
decision (+186.106724); the largest improvement is class780/seed80 on the same
set (−2.970177). No class, seed or extreme row is deleted or used to refit a detector.

### Conditional headroom and the unfavorable secondary family

Stage15 is the modal oracle in 25/27 sets; 14/27 have a completely fixed oracle
across all 64 class/seed units. These 14 sets have zero empirical oracle-switching
headroom, up to scalar cancellation tolerance. The empirical best fixed choice
for **mean absolute risk** is stage15 in all 27 sets, with equal-set mean
absolute headroom 0.019874962. Optimizing normalized regret separately gives
mean headroom 0.045282708. These test-best-fixed choices are diagnostics, not
new deployable policies. A modal oracle and the minimum-expected-risk candidate
need not coincide.

Five sets have descriptive `oracle-majority < 0.75` (320 units): attention-output
W4/W6, MLP-up W4/W6 and adaLN-conditioning W6. Their outcome-defined subset is
exploratory, not a new primary PASS criterion; the full grid stays primary.
Per-set majority rates, entropies, headroom and subset results are linked in
[the P0 audit](BASELINE_AUDIT.md).

The original cross-group secondary summary favors Constant: Hybrid−Constant
normalized regret is +0.0007820371, 95% CI [+0.0000246697, +0.0017184796].
It remains separate as `source_summary_only`: public membership and frozen
Constant configuration support the primary 27-set reconstruction, not the
secondary grid. Secondary source values are checked, not presented as newly
reconstructed choices or pooled into the primary mean.

### P1: stage prior and aligned geometry controls

All scores use the original test residual energy and fixed beta
0.6735229330314306. Stage-only uses
`E * exp(beta * mean_source_cells_at_stage(log G))`, with 27 source cells per
stage weighted equally. It is not the negative-stage-position rule in P0.
Actual, donor and isotropic scores use aligned historical control coefficients;
no G or beta is refitted. Actual-source and frozen Candidate-B differ only in
preserved source decimal representations and choose identically here.

| Geometry score | Mean absolute regret | Mean normalized regret |
|---|---:|---:|
| Actual | 0.186483 | 0.043135 |
| Donor | 0.054314 | 0.053453 |
| Isotropic | 0.264475 | 0.090312 |
| Stage-only | 0.187148 | 0.044581 |

| Comparison | Absolute difference [95% CI] | Normalized difference [95% CI] |
|---|---:|---:|
| Actual − Stage-only | −0.000665 [−0.003719, +0.003077] | −0.001446 [−0.006578, +0.004110] |
| Actual − Donor | +0.132169 [−0.012878, +0.361158] | −0.010318 [−0.023486, +0.002685] |
| Donor − Isotropic | −0.210161 [−0.439855, −0.052315] | −0.036859 [−0.046975, −0.026537] |
| Actual − Isotropic | −0.077992 [−0.127015, −0.039094] | −0.047177 [−0.058209, −0.036217] |

Actual geometry is not demonstrably superior to stage-only or donor scores under
these intervals. Donor geometry outperforms isotropic geometry, in both regret
units, so structured directions can carry decision-relevant signal beyond the
isotropic comparison. This does not show that the signal specifically requires
originating-state residual pairing, or identify its unique causal source.
See [PAIRING_ABLATION.md](PAIRING_ABLATION.md) for matched support, source paths,
stage definitions, scalar-only verification limits and detailed tables.

## Negative result: one-step exception detection

E1-Tail0 concluded `E1TAIL0_NO_USEFUL_EXCEPTION_SIGNAL`.
There were 79 relative exceptions in 1,728 units. The detector had
TP/FP/FN/TN = 3/474/76/1175: recall 3.80%, FPR 28.74%, AP 0.0519 and ROC-AUC
0.5937. It missed 76 of 79 exceptions. Favorable isolated point estimates do not
override this failed preregistered detection gate. The detector is not packaged
as a validated protective method, and no threshold was refitted here.

## What remains open

The evidence supports residual–geometry coupling and the original prospective
native-dtype improvement over Energy under the tested scope. The retrospective
P0/P1 audit does not establish additional adaptive utility beyond strong
static/stage priors, or the necessity of originating-state pairing. A failed
simple exception detector further constrains method claims.
External architecture generalization, actual packed execution
utility and decoded/task quality are unestablished. FP32 is not ground truth;
dtype agreement is not a required proof of native-FP16 predictor validity.

Raw terminal/state tensors are deliberately excluded. Public scalar checks
cannot certify raw-output generation, leakage audits or tangent validity anew.
