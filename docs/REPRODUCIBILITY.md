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

## Installation contracts

Supported Python versions are 3.11 and 3.12. Runtime dependencies are
NumPy >=1.26,<2 and PyTorch >=2.11,<2.12. The `test` extra adds pytest; `dev`
adds test, build, lint and YAML-validation tools. The optional `dit` extra adds
Diffusers 0.38.0. Neither core import nor the example loads pretrained weights.

The [CPU constraints](../requirements/cpu-constraints.txt) pin the Linux x86_64
dependency set used by [CI](../.github/workflows/ci.yml). Install PyTorch first
from its [official CPU index](https://download.pytorch.org/whl/cpu/torch/), then
extras from PyPI, as shown in the [README](../README.md). Python 3.13 and NumPy 2
are outside this support contract. These CPU constraints do not prescribe a
CUDA build for existing research environments.

### Editable development

In a fresh virtual environment, after installing the constrained CPU PyTorch:

```bash
python -m pip install -c requirements/cpu-constraints.txt build setuptools wheel
python -m pip install --no-build-isolation -c requirements/cpu-constraints.txt -e ".[dev,dit]"
python -m pip check
python -m pytest -q
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
A local equivalent run is not evidence of a completed hosted Actions run.

### Existing pinned offline environment

When dependencies and build tools already exist, use the README's
`--no-index --no-deps --no-build-isolation` command. It cannot provision an empty
offline machine. A system-site-packages virtual environment reuses dependencies
and is not an isolated dependency-install test. The optional Diffusers `kernels`
extension is not used by the example and is not required.

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
