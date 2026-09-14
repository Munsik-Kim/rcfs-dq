# P1: stage prior and residual–state pairing

This is a retrospective CPU-only reanalysis of existing public scalar evidence,
with identified historical scalar control coefficients in a separate derived
layer. Model calls = 0; GPU calls = 0; new model observations = 0;
predictor refits = 0; new prospective confirmation = 0.
The original frozen coefficients and scientific evidence are unchanged.

The question is whether originating-state residual pairing adds decision-relevant
information beyond stage structure or a generic structured direction. This audit
does not estimate mutual information, prove pairing necessary, or identify a
unique causal source of utility.

## Scores and aligned support

The stage-only coefficient uses all 27 source cells at each stage equally:

```text
a_stage = exp(beta * mean_source_cells_at_stage(log G))
stage_only_score = E * a_stage
beta = 0.6735229330314306
```

Coefficients for stage indices 4, 10 and 15 are 8.547380373194017,
6.739903175686341 and 1.1327646313220412. This is a retrospective source-only
ablation, not a rule preregistered before the historical test outcomes existed.
It is distinct from P0's negative-stage-position rule, which does not use E.

The three geometry scores are `E * G_direction**beta` for aligned actual,
shuffled donor and isotropic directions. Isotropic G is the arithmetic mean
of two draw-level squared-gain ratios of sums, not a mean of receiver-level
ratios, an amplitude gain or a universal `1/d` null.

The [source certificate](../analysis/pairing_ablation/source_alignment_certificate.json)
and [cell alignment](../analysis/pairing_ablation/source_alignment.csv) identify
historical scientific commits `7f21fdc1a5955e7d42ae0229a8db5056a2329027`
and `f55d9777907b8aafad9c2adacf46e7c4ad5ec5af`, their source hashes and contracts.
The certificate records all 81 cells with 22–24 common-valid receiver rows per
cell, eight represented discovery classes, matching fixed alpha and receiver
sets across actual/two isotropic/donor directions. Target seeds 1–3 use distinct
donor seeds 16–18 within the same class and cell; directions are norm-matched.

This provenance certificate reports earlier read-only source-scalar checks.
The public verifier checks its consistency with the supplied coefficients and
recalculates decisions from those scalars; it does not rerun the historical
source audit, regenerate raw directions or certify tangent validity anew.
The historical geometry path is `ideal_fp32_central`, batch eight and terminal
latent endpoint. It is separate from the native-FP16 decision target; FP32 is
not ground truth.

The original frozen G equals the E1-Pred1 transfer table. Ten of 81 coefficients
differ from E1-D1 CSV decimals at the final digits (maximum absolute difference
`1.4210854715202004e-14`). Both are preserved: `candidate_B` uses frozen JSON,
`aligned_actual_source` uses the aligned historical scalar. Their choices agree
on all 1,728 units. No source value is replaced to manufacture exact equality.

## Results

The same 27 sets, 16 classes and four seeds yield 1,728/1,728 resolved units for
every score. Means equally average sets, seeds and then classes. Raw tails are
descriptive. Source-native latent-severity units and range-normalized regret
remain separate.

| Geometry score | Mean absolute regret | Mean normalized regret |
|---|---:|---:|
| Actual | 0.186483 | 0.043135 |
| Donor | 0.054314 | 0.053453 |
| Isotropic | 0.264475 | 0.090312 |
| Stage-only | 0.187148 | 0.044581 |

Paired differences use 10,000 shared class-block resamples, PCG64(2026091101),
linear 0.025/0.975 percentiles. Negative means the left score has lower regret.

| Comparison | Absolute difference [95% CI] | Normalized difference [95% CI] |
|---|---:|---:|
| Actual − Stage-only | −0.000665 [−0.003719, +0.003077] | −0.001446 [−0.006578, +0.004110] |
| Actual − Donor | +0.132169 [−0.012878, +0.361158] | −0.010318 [−0.023486, +0.002685] |
| Donor − Isotropic | −0.210161 [−0.439855, −0.052315] | −0.036859 [−0.046975, −0.026537] |
| Actual − Isotropic | −0.077992 [−0.127015, −0.039094] | −0.047177 [−0.058209, −0.036217] |

Actual−stage changes 78 choices, helping 40 and hurting 38; actual−donor changes
210, helping 114 and hurting 96. Donor−isotropic changes 173, helping 131 and
hurting 42; actual−isotropic changes 248, helping 179 and hurting 69.
Normalized regret improves at class level in 9/16, 10/16, 15/16 and 16/16,
respectively. A change count is not an effect-size or tail guarantee.

Actual geometry does not establish superiority over the stage-only or donor
score: both intervals cross zero in both regret units. Actual−donor absolute
regret is positive even though its normalized point is negative. Donor and
actual both improve over isotropic geometry. Structured geometry can therefore
carry useful signal in this task, but its additional originating-state-specific
pairing value remains unestablished. Crossing zero does not prove equivalence.

The historical class85/seed82 MLP-up W4 extreme remains: Candidate-B and aligned
actual select stage4 with regret 186.507995. No outlier or class is discarded;
largest changes and leave-one-class-out influence are diagnostic only.

## Stage, endpoint and interpretation boundaries

Historical geometry perturbs the full-precision post-update state `x_(k+1)`
and continues at `k+1`, so its remaining horizon at K=20 is `K-k-1` (15/9/4).
The natural intervention includes update k and has `K-k` suffix steps (16/10/5).
These observables and their computation costs are not interchangeable.

[Stage observables](../analysis/pairing_ablation/stage_observables.csv) preserve
historical actual/donor/isotropic magnitudes, per-cell log contrasts and later
native test E/Y as distinct populations. Stage10's mean historical state
contrast is not a Jacobian-gain theorem. Scalar ratios and directions do not
establish allocation utility, architecture-general validity or causal necessity.

## Reproduce and inspect

```bash
python scripts/audit_pairing_ablation.py --check
python scripts/verify_baseline_audit.py --pairing --check
python scripts/verify_public_audits.py
```

The main P1 audit reuses validated P0 input loading and public decision primitives.
Its independent NumPy implementation does not import main score, aggregation or
gate functions. It derives coefficients, choices, regret, class means and paired
bootstrap from scalar inputs. Discrete identity is exact; floating agreement uses
`abs(main-ref) <= 1e-12 + 1e-10 * abs(ref)`.

- [Score results and tails](../analysis/pairing_ablation/score_results.csv)
- [All paired contrasts](../analysis/pairing_ablation/paired_comparisons.csv)
- [Class endpoints](../analysis/pairing_ablation/class_endpoints.csv)
- [Decision changes](../analysis/pairing_ablation/decision_changes.csv)
- [Largest changes](../analysis/pairing_ablation/largest_policy_changes.csv)
- [Class influence](../analysis/pairing_ablation/leave_one_class_out.csv)
- [Per-analysis manifest](../analysis/pairing_ablation/audit_manifest.json)
  and [independent verification](../analysis/pairing_ablation/independent_verification.json)

The [roadmap](ROADMAP.md) prioritizes a separately frozen stage-neutral
prospective test. P1 itself is not that test and triggers no new model execution.
