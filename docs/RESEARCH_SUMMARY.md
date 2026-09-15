# Research summary: local information is not natural-scale utility

The original question was whether actual quantization residuals align with
future-sensitive diffusion directions and whether this structure changes a
decision in the right direction. Five layers answer different parts of it.

**Strong-baseline audit.** Candidate-B/Hybrid improved over Energy in the
original prospective native-FP16 matched-cost task. Retrospective P0/P1 retained
that result but exposed limited oracle headroom and competitive Constant and
latest-stage policies. Additional adaptive utility over strong static/stage
priors and the unique value of originating-state pairing were not established.
The E1-Tail0 simple exception detector also failed.

**Numerical qualification.** J0 v1 failed: 157/432 model-Jv consistency rows.
NR0 identified operational FP32 scheduler/central-difference roundoff as the
dominant explanation, not a reason to rewrite v1 as PASS. Scheduler-only FP64
reconstruction is not FP64 model execution. NR1 and subsequent tangent/radius
review separated representation, resolution, tangent stability and finite-radius
adequacy. Affine algebraic closure alone is not scientific validation.

**Local measurement.** Locked V3 found substantial Full prediction gains over
No-J and frozen Scalar within the primary qualified small-radius scope. Its
target was +1/128 and derivative ±1/64. Full consumed additional directional
evaluations. Actual-local R_J was lower than donor/isotropic, not uniquely larger.
No application certificate followed.

**Natural-error bridge.** BRIDGE0 reused the full error after one stage4 W4
intervention and restored-model continuation. Full won on 178/192 available
probe5→6/probe11→12 transitions, but large MLP-down/probe11 losses produced a
class-balanced mean absolute Full−Scalar difference of +0.0102570311,
95% CI [-0.0493295519, 0.083345637]. Neither superiority nor equivalence was
established. The 96 probe16→17 natural targets were not stored.

**Limited closeout.** Local predictive information != stable natural-scale
utility. `RCFS_DQ_LOCAL_MECHANISM_TRACK_CLOSED_AS_LIMITED` is a project-scope
decision, not a replacement scientific gate. Small-radius information survives;
stable natural-scale mean benefit over strong Scalar does not follow. Changing
alpha, gamma or subgroup routing would create a new unvalidated question.

Useful assets remain: residual/intervention primitives, finite-horizon controls,
regret/headroom audits, reference verification, hash provenance, replay and tail
accounting. [Future priorities](ROADMAP.md) are strong-baseline/headroom research
and scoped artifact qualification, not automatic rescue.
[Exact results](RESULTS.md) and [limitations](LIMITATIONS.md) bound the claims.
