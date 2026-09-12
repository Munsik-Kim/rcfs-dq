# Method and conventions

## Residual and finite-horizon geometry

At stage k, a single weight-quantized update produces a state-space residual:

$$\eta_k=F_k^{(q)}(x_k)-F_k(x_k).$$

This is not the weight residual Q(W)−W. The future continuation from the
post-update state defines the directional response. Conceptually its amplitude
gain is ||J eta|| / ||eta||. The **squared gain** used in the frozen source
artifacts is the ratio of sums

$$\mathrm{LCG}=\frac{\sum_i\|v_i\|_2^2}{\sum_i\|u_i\|_2^2},\qquad
v_i=\frac{\Phi(x_i+\alpha u_i)-\Phi(x_i-\alpha u_i)}{2\alpha}.$$

It is not a mean of row-wise ratios. Finite differences require separate
tangent-support and realized-injection checks; this package does not assert
their validity for an arbitrary model or amplitude. `finite_horizon_response`
returns the realized positive/negative injection as well as the response.

Source controls include norm-matched isotropic directions and shuffled donor
residuals. Sign- and channel-randomized generators are reusable secondary
utilities. Comparisons must share valid rows. Isotropic reference gain averages
the draw-level LCGs. The state contrast is

$$L_{\mathrm{state}}=\log(\mathrm{LCG}_{\mathrm{actual}}+10^{-30})
-\log(\mathrm{LCG}_{\mathrm{shuffled}}+10^{-30}).$$

The small log floor is algebraic, not an empirical numerical-noise estimate.

Control inputs must be nonempty finite floating-point tensors. Donors must have
exactly the target shape; no broadcasting is performed. Norm-matched outputs
have independent contiguous storage, including when the donor is strided.
A valid zero target returns zeros; a nonzero target with a zero donor is invalid.
Channel shuffling requires a channel dimension. Unsupported norm ranges raise
`ValueError`, and representable norm matching is checked to relative error 1e-6.

LCG keeps sum-of-squares arithmetic in the ordinary float64 range. At extreme
ranges it scales norms and combines binary exponents before forming the ratio.
A zero response with positive direction energy has gain zero. A true zero
denominator or a positive ratio outside representable float64 range raises
`ValueError`; no epsilon replaces a denominator.

## Frozen score, not a fitting interface

$$S_B(c)=E(c)G_c^\beta,\qquad \beta=0.6735229330314306.$$

E is the source one-update state residual energy. G is the frozen **per-cell
actual squared-gain ratio-of-sums**, not a group-stage geometric-mean substitute.
The unchanged values are in `configs/frozen_predictor.json`. The source log
calibration intercept is 0.44375809726952603; it does not affect selection.
The public score class has no fitting API.

M3 comprises `mlp_down_all_blocks` W4 at stages 4/10/15. L78 is its Core81
complement. For low-risk selection, Hybrid prioritizes L78 over protected M3;
within M3 it uses energy, and within L78 Candidate-B. On the primary same-group,
same-bit sets, this produces the same selections as Candidate-B in the supplied
evidence. It is not an independent extra successful predictor.

The source Constant uses development data only: seed median per class/cell,
then the median across eight source classes. Its frozen choices do not adapt to
the test class or seed.

## Intervention and arithmetic

Quantization is symmetric absmax, per output channel, using signed integer
range [−2^(q−1), 2^(q−1)−1], float32 scaling/rounding and dequantization to the
original weight dtype. It is **fake quantization**, not a packed execution kernel.
The named module scope is explicit. Weights are restored even on exceptions.

For q < 16, weights, FP32 operands, nonzero-channel scales and returned outputs
must be representable and finite. Nonzero values lost during FP32 conversion,
zero-underflowed scales and overflowing outputs raise `ValueError`; scales are
not repaired with epsilon. All-zero channels retain the unit-scale convention.
The identity modes (`None` or 16) clone finite inputs in their original dtype
without FP32 conversion. Empty, integer and complex weights are unsupported.

A natural intervention changes one sampler update, then restores weights for
the unchanged native continuation. There is no state reset or fitted correction.
The caller must preserve conditioning, scheduler state and RNG semantics.

The matched-cost source energy and terminal severity use float32 subtraction
after separately promoting the absolute native endpoints, followed by FP64
squared-norm reduction. `endpoint_difference(arithmetic='source_native')`
preserves that convention. The generic `fp64` option is explicit and is **not**
a drop-in numerical substitution for source-native reproduction.

## Exact-cost decisions

For n selected weight parameters at q bits:

$$C_{\mathrm{packed}}=nq,\qquad B=n(16-q).$$

Primary sets share group, bitwidth and **exact** saving bits; stage is the
candidate choice. The included 81 candidates form 27 three-stage sets. Tensor
shape products are checked against n. No percentage-tolerance bins are used.
Secondary cross-group equivalence must remain separate from primary estimates.

$$R_p=Y(\hat c_p)-\min_{c\in\mathcal C}Y(c).$$

Ties use lower stage, canonical group order, higher bitwidth, then cell ID.
Normalized regret divides by the within-set risk range only when that range
exceeds 1e-12 × max(max Y, 1). Otherwise it is undefined; absolute regret remains.

Primary aggregation is equal set mean within class/seed, four-seed mean within
class, then equal mean across 16 classes. Paired bootstrap uses shared class
indices: PCG64(2026091101), 10,000 draws, linear quantiles. Seed-median maps and
secondary sets are different estimands and are not pooled into this result.

Neither scalar prediction nor decision utility establishes joint composability,
an allocation method, decoded quality, or an error-free FP32 reference.
