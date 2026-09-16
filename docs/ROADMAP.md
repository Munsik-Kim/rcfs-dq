# Roadmap after limited closeout

The local-mechanism-to-natural-utility track is closed as limited. Its historical
decisions, negative observations and missing data remain intact. No default
alpha tuning, gamma refitting, MLP rescue or missing-probe completion is planned.
The following are priorities for separate review, not execution authorizations.

## A. External measurement generalization — OPTIONAL / RESEARCH

Only with a clear publication objective, test the already-frozen measurement
question on a genuinely new model, scheduler or quantizer. Preregister strong
baselines, input separation, error-scale targets, endpoints and tail handling.
Do not assume either universal Jacobian importance or universal uselessness.

## B. Adaptive-headroom / strong-baseline audit — PRIORITY RESEARCH

Measure whether a task offers enough adaptive headroom to justify complexity:
best fixed expected loss minus expected per-input oracle loss. Report oracle
diversity, absolute and normalized headroom, tails, and strong fixed and
zero-parameter heuristics before crediting an adaptive policy. A test-best-fixed
diagnostic is not a deployable policy learned without test outcomes.
Do not assume the stage degeneracy found here characterizes every benchmark.

## C. ModelDiff Guard / quant-qualify — PRIORITY PRODUCT

Reuse artifact identity, provenance, workload binding, replay, resource
measurement and catastrophic-tail checks to qualify an existing quantized
artifact for an explicitly observed workload and hardware scope. Require
decoded/task-quality and resource criteria, conversion/setup accounting and
regression checks. This is not a new RCFS optimizer or universal safety
certification. No acceptance of a deployment artifact is established here.

J1 RTN-vs-GPTQ, P2 stage-neutral decisions and the probe16 natural supplement
are **DEFERRED / NOT CURRENTLY AUTHORIZED**, not scientifically disproven.
Allocation and new scientific experiments are not automatic follow-ups.
