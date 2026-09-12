# Reproducibility: three distinct scopes

1. **Core tests:** synthetic algebra, quantization, restoration, controls,
   score freezing, exact-cost decisions, ties, undefined ranges, paired
   bootstrap and independent regret checks.
2. **Compact example:** tiny random CPU DiT, four DDIM steps, one stage-local
   intervention, residual extraction and finite-horizon control calculations.
   No model download or GPU is needed. This tests plumbing, not research claims.
3. **Public evidence check:** SHA/size inventory; 81 candidate costs; 5,184
   scalar candidate records; frozen scores; 1,728 decisions per policy;
   independent oracle/regret checks; class-block mean/CI reproduction for the
   four primary policies; discovery contrast arithmetic; Tail0 detection counts.

The public evidence check does not independently regenerate terminal risk from
raw tensors, rerun discovery/confirmation, or replay the entire internal science
pipeline. Secondary summary tables are preserved as source records; the compact
decision recomputation targets the 27 primary sets only.

## Offline installation and checks

The inspected source environment is Python 3.11.15, NumPy 1.26.4,
PyTorch 2.11.0+cu128, Diffusers 0.38.0, pytest 9.1.1, setuptools 70.2.0 and
wheel 0.47.0. No package/driver update is part of this export.

Use an environment where these dependencies already exist:

```bash
python -m pip install --no-index --no-deps --no-build-isolation .
python -c "import rcfs_dq; print(rcfs_dq.__version__)"
python scripts/verify_evidence.py
python examples/compact_dit.py
python -m pytest -q
```

A clean temporary package copy and a separate virtual environment reusing
installed dependencies are the release smoke scope. That is not a claim that a
fully offline empty machine can acquire dependencies. The package does not
download them automatically in the commands above. The optional Diffusers
`kernels` extension is not used by the example.

## Historical experiment boundary

The model identifier is `facebook/DiT-XL-2-256`, revision
`eab87f77abd5aef071a632f08807fbaab0b704d0`, checkpoint SHA-256
`e592d64df5a579691e65d2b245641a00bb070b652e2c5ca775cce20a729ce9d9`.
The source used deterministic 20-step DDIM, eta=0, CFG off, math SDPA, TF32 off,
native mixed-internal FP16 arithmetic and restored stage-local fake weights.
The matched-cost source used prefix batch 16 and event/suffix batch 8.
`configs/reference_dit.json` records this boundary; it is not an execution
authorization or a replacement full-run lock.

The reusable `DDIMPath` calls the installed scheduler/prediction path and exposes
cropping explicitly. It does not claim bitwise equivalence between the tiny CPU
example and historical GPU experiments, or between different batch layouts.
Applications must validate their own scheduler, state transport and module scope.

## Export-only changes and independent checks

The source repository is unchanged. This export removes phase dependencies and
adds explicit empty/duplicate scope, nonfinite and invalid-bit guards. The
control norm-matching helper clones a donor before scaling, so caller-owned
directions are not mutated. Probe injections and returned endpoints are
snapshotted before a callback can reuse a buffer. Tests cover these changes.

Export preparation also compared selected unchanged source functions on
synthetic CPU inputs: 20 quantizer and 16 control cases were bitwise identical;
10 gain/contrast cases and five class-CI fields matched; eight tiny DiT/DDIM
stage updates (baseline and quantized) matched source stage methods bitwise.
This is engineering evidence only, not a historical GPU/full-model replay.

Independent regret is implemented separately from selection/aggregation; sharing
file readers, registries and hash utilities is explicit. Numerical comparisons
use abs(main−ref) <= 1e-12 + 1e-10 × abs(ref). SHA and bitwise checks are separate.
The supplied evidence values are source-derived—not synthetic model outputs.

The spatially shuffled direction in the compact example is a **toy donor**, not
an independently sampled scientific state-shuffle control. Model calls and toy
results must not be counted as fresh RCFS-DQ observations.
