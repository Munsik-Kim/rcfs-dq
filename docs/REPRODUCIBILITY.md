# Reproducibility: distinct scopes

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
4. **P0/P1 derived audit:** retrospective CPU-only selections, static/stage
   baselines, headroom and aligned geometry ablations. Independent NumPy paths
   recalculate regret and class-block intervals, with a separate audit inventory.
   Model calls = 0; GPU calls = 0; new model observations = 0;
   predictor refits = 0; new prospective confirmation = 0.
5. **Closeout scalar reference:** separately inventoried V3/BRIDGE0 scalar rows
   and historical decisions. Recompute coverage, C/D rules, inherited R/N
   flags, normalization, class aggregation, paired bootstrap, tails and influence.
   Independent NumPy code imports no research prediction/gate implementation.
   Stored norms/errors remain inputs, not independently regenerated vectors.

The public evidence check does not independently regenerate terminal risk from
raw tensors, rerun discovery/confirmation, or replay the entire internal science
pipeline. Secondary summary tables are preserved as source records; the compact
decision recomputation targets the 27 primary sets only.

The original layer remains under
[PUBLIC_EVIDENCE_MANIFEST.json](../PUBLIC_EVIDENCE_MANIFEST.json): `evidence/`,
frozen `configs/` and `figures/` retain their original bytes.
[PUBLIC_AUDIT_MANIFEST.json](../PUBLIC_AUDIT_MANIFEST.json) separately records
derived `analysis/baseline_audit/` and `analysis/pairing_ablation/` payloads and
their inputs/implementation. The P1 source-alignment certificate is a historical
scalar provenance receipt, not a newly reproduced raw-tensor experiment.

## Read-only scalar and audit verification

From the repository with the pinned dependencies:

```bash
python scripts/verify_evidence.py
python -O scripts/verify_evidence.py
python scripts/audit_baselines.py --check
python scripts/audit_pairing_ablation.py --check
python scripts/verify_baseline_audit.py --check
python scripts/verify_baseline_audit.py --pairing --check
python scripts/verify_public_audits.py
python -O scripts/verify_public_audits.py
python scripts/verify_closeout_evidence.py
python -O scripts/verify_closeout_evidence.py
```

Evidence validation uses explicit failures, not removable Python assertions;
optimized Python must enforce the same scientific/data checks. Audit `--check`
modes recompute into memory and compare the committed payloads without replacing
them. The independent verifier derives results from scalar inputs, not README
numbers or main summary booleans. Main scripts support `--output` to a new or
empty directory when a separate scalar export is needed. Original evidence is
not an output destination.

P0 checks candidate/cost coverage, frozen scores, aliases, regret, aggregation
and paired bootstrap. P1 checks historical coefficient/support consistency and
actual/stage/donor/isotropic decisions. The certificate's underlying raw state,
residual and response tensors are not included. Hash agreement cannot establish
that the historical model outputs or tangent conditions were independently
regenerated. See [P0 definitions](BASELINE_AUDIT.md) and [P1 scope](PAIRING_ABLATION.md).

## Closeout scalar scope

[PUBLIC_CLOSEOUT_MANIFEST.json](../PUBLIC_CLOSEOUT_MANIFEST.json) is a third,
separate provenance layer. Source decisions/tables retain their original bytes;
the public scalar reference cannot rerun historical raw-verifier receipts.
V3 uses a four-seed median, nine-spec/eight-class estimator at small radius;
BRIDGE0 uses a four-seed mean, six available specs/eight classes at natural scale.
Neither scalar arithmetic nor recorded SHA identity recreates model predictions.

## Installation contracts

Supported Python versions are 3.11 and 3.12. Runtime dependencies are
NumPy >=1.26,<2, PyTorch >=2.14,<2.15 and setuptools >=83,<84. Builds use the
same setuptools range. The direct runtime constraint also tightens Torch's
transitive setuptools requirement for wheel installs outside our CPU constraints.
The `test` extra adds pytest, build and YAML-validation tools; `dev`
adds test, build, lint and YAML-validation tools. The optional `dit` extra adds
Diffusers 0.38.0. Neither core import nor the example loads pretrained weights.

The [CPU constraints](../requirements/cpu-constraints.txt) pin the Linux x86_64
dependency set used by [CI](../.github/workflows/ci.yml). Install the constrained
build tools from PyPI, then PyTorch from its
[official CPU index](https://download.pytorch.org/whl/cpu/torch/), then project extras.
The build tools must already satisfy the Torch dependency before using the
separate CPU index. Python 3.13 and NumPy 2
are outside this support contract. These CPU constraints do not prescribe a
CUDA build for existing research environments.

### Editable development

In a fresh virtual environment:

```bash
python -m pip install -c requirements/cpu-constraints.txt build setuptools wheel
python -m pip install -c requirements/cpu-constraints.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -c requirements/cpu-constraints.txt -e ".[dev,dit]"
python -m pip check
python -m pytest -q
python scripts/verify_evidence.py
python -O scripts/verify_evidence.py
python scripts/verify_public_audits.py
python -m ruff check .
python -m ruff format --check .
HF_HUB_OFFLINE=1 python examples/compact_dit.py
```

### Wheel installation

Use a separate virtual environment with the same dependencies; do not leave an
editable install or `PYTHONPATH` pointing at this checkout:

```bash
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/rcfs_dq-0.1.1-py3-none-any.whl
python -m pip check
```

From outside the checkout, check the installed package with isolated import:

```bash
python -I -c 'import pathlib, sysconfig, rcfs_dq; p = pathlib.Path(rcfs_dq.__file__).resolve(); assert p.is_relative_to(pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()); print(p, rcfs_dq.__version__)'
```

The wheel supplies reusable Python code. Evidence, examples and configuration
files are repository payloads; run their checks from the repository after the
installed import check. GitHub Actions defines both supported Python versions,
wheel import, tests, scalar verification, offline tiny DiT, lint and hygiene.
A separate audit check remains model-free; the tiny random DiT makes synthetic
CPU model calls only and must not be counted as a P0/P1 observation.
A local equivalent run is not evidence of a completed hosted Actions run.

### Existing pinned offline environment

When the current supported dependencies and build tools already exist, use the
offline command below. It cannot provision an empty
offline machine. A system-site-packages virtual environment reuses dependencies
and is not an isolated dependency-install test. The optional Diffusers `kernels`
extension is not used by the example and is not required.
`--no-deps` and `--no-build-isolation` do not upgrade an old environment. Check
the current constraints and build-tool version first; an old historical research
environment is not a qualified current package environment.

```bash
python -m pip install --no-index --no-deps --no-build-isolation .
python -m pip check
```

### Dependency security update

The current CPU contract pins torch 2.14.0+cpu and setuptools 83.0.0. Both the
runtime metadata and isolated-build requirements exclude the previously flagged
ranges; changing only the constraints would leave alternate installation paths.
The tracked advisories are [PyTorch JIT](https://github.com/advisories/GHSA-rrmf-rvhw-rf47),
setuptools [path traversal](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf)
and [Unicode exclusion](https://github.com/advisories/GHSA-h35f-9h28-mq5c).
See the [upstream setuptools release record](https://setuptools.pypa.io/en/latest/history.html#v83-0-0).

The public API does not load untrusted models or expose JIT compilation or
PackageIndex downloads. Dependency remediation is not a claim that an exploit
was reachable through an RCFS-DQ API, or that arbitrary model/code inputs are safe.
The initially considered torch 2.13.0+cpu still failed the bare-container JIT
regression in an isolated subprocess; it was not accepted on version metadata
alone. The 2.14 release includes the [upstream JIT fix](https://github.com/pytorch/pytorch/commit/b90c94991cdf8b87c8f7439f79518e0ef2c4ca4f).
Security regression tests exercise the supported build backend, bare local
container annotations and ordinary typed JIT/file-exclusion behavior. They do
not establish that every malformed JIT input is handled safely. These are
software tests, not new science.
Direct legacy distutils packaging is outside this build contract; its helper
is not certified by the setuptools-backend exclusion check.
Historical environment records and all scientific scalar payloads remain unchanged.

## Historical experiment boundary

The model identifier is `facebook/DiT-XL-2-256`, revision
`eab87f77abd5aef071a632f08807fbaab0b704d0`, checkpoint SHA-256
`e592d64df5a579691e65d2b245641a00bb070b652e2c5ca775cce20a729ce9d9`.
The source environment used Python 3.11.15, NumPy 1.26.4, PyTorch 2.11.0+cu128
and Diffusers 0.38.0. It used deterministic 20-step DDIM, eta=0, CFG off, math SDPA, TF32 off,
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
