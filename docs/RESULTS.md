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

## Prospective native-FP16 matched-cost decisions

The fresh 16-class × 4-seed panel contains 1,728 class/seed/set decision units
across 27 same-group/same-bit exact-cost stage-selection sets. Uncertainty uses
16 class blocks, not 1,728 independent population samples.

| Frozen policy | Mean absolute regret | Mean normalized regret |
|---|---:|---:|
| Energy | 0.292808 | 0.185153 |
| Source Constant | 0.210804 | 0.046017 |
| Candidate-B | 0.186483 | 0.043135 |
| Hybrid | 0.186483 | 0.043135 |

Hybrid minus Energy normalized regret was −0.142018, with two-sided 95% class
bootstrap interval [−0.152157, −0.131562]. Hybrid minus Constant was −0.002882,
interval [−0.007260, 0.001869]. Thus the frozen geometry policy improved over
Energy, but **strict normalized-regret superiority over the strong source
Constant was not established**. Candidate frequency/diversity records must be
read alongside regret: a nearly fixed best stage offers limited adaptive headroom.

These are theoretical packed-storage-matched, single-intervention decisions.
They are not measured memory/latency savings or a full allocation result.

## Negative result: one-step exception detection

E1-Tail0 concluded `E1TAIL0_NO_USEFUL_EXCEPTION_SIGNAL`.
There were 79 relative exceptions in 1,728 units. The detector had
TP/FP/FN/TN = 3/474/76/1175: recall 3.80%, FPR 28.74%, AP 0.0519 and ROC-AUC
0.5937. It missed 76 of 79 exceptions. Favorable isolated point estimates do not
override this failed preregistered detection gate. The detector is not packaged
as a validated protective method, and no threshold was refitted here.

## What remains open

The evidence supports residual–geometry coupling and limited prospective
native-dtype decision value over Energy under the tested scope. A competitive
constant baseline and a failed simple exception detector constrain adaptive
method claims. External architecture generalization, actual packed execution
utility and decoded/task quality are unestablished. FP32 is not ground truth;
dtype agreement is not a required proof of native-FP16 predictor validity.

Raw terminal/state tensors are deliberately excluded. Public scalar checks
cannot certify raw-output generation, leakage audits or tangent validity anew.
