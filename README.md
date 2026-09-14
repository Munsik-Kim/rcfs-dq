# RCFS-DQ

Residual-conditioned finite-horizon sensitivity for diffusion quantization.

**Actual quantization residual → future-sensitive geometry → decision audit.**

RCFS-DQ asks whether actual quantization residuals occupy future-sensitive
directions non-randomly, and whether that structure improves quantization
decisions. It does not claim that finite-time sensitivity itself is new.

## Evidence in brief

- Core81 discovery: 81/81 cells met the locked tangent-support criterion.
- Independent confirmation: positive-family mean 1.9092, one-sided exact
  p = 0.00390625; expected sign in 9/9 cells, stricter replication in 8/9.
- On the original prospective native-FP16 panel, frozen Candidate-B/Hybrid
  improved substantially over Energy. P0/P1 add a **retrospective CPU-only
  reanalysis of existing public scalar evidence**, not new confirmation.
- Strong Constant and zero-forward stage priors are competitive. Additional
  adaptive utility beyond them is **not established**; latest-stage has lower
  mean absolute regret. Actual geometry is not demonstrably superior to donor
  geometry or a stage-only score; donor outperforms isotropic geometry.
- Negative result: the simple one-step exception detector failed
  (recall 3.80%, FPR 28.74%, AP 0.0519). E1-Tail0 remains a negative result.

| Policy | Mean absolute regret | Mean normalized regret |
|---|---:|---:|
| Energy | 0.292808 | 0.185153 |
| Source Constant | 0.210804 | 0.046017 |
| Latest-stage | 0.019875 | 0.052227 |
| Candidate-B | 0.186483 | 0.043135 |

All policies use the same 1,728 decision units and 16 class blocks. Hybrid
matches Candidate-B; Pure-G and negative-stage position match Latest-stage on
every unit. P0/P1 have model calls = 0, GPU calls = 0, new model observations = 0,
predictor refits = 0 and new prospective confirmation = 0.
The original prospective primary endpoint and recorded decision are unchanged;
new static comparisons and absolute-risk limitations are retrospective additions.

See [results](docs/RESULTS.md) for paired intervals, absolute-risk tails and
headroom, [P0](docs/BASELINE_AUDIT.md) and [P1](docs/PAIRING_ABLATION.md) for audits, and
[method](docs/METHOD.md) for exact definitions.

## Reproduce the public checks

Supported: Python 3.11–3.12, NumPy 1.26.x and PyTorch 2.11.x.
From this repository, create a CPU environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/cpu-constraints.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/cpu-constraints.txt ".[test]"
python scripts/verify_evidence.py
python -O scripts/verify_evidence.py
python scripts/verify_public_audits.py
python -m pytest -q
```

Optional tiny random DiT check (CPU, no checkpoint download):

```bash
python -m pip install -c requirements/cpu-constraints.txt ".[dit]"
HF_HUB_OFFLINE=1 python examples/compact_dit.py
```

Already have the pinned dependencies? Reinstall offline:

```bash
python -m pip install --no-index --no-deps --no-build-isolation .
```

See [reproducibility](docs/REPRODUCIBILITY.md) for development and wheel checks.
The evidence check recomputes frozen scores, decisions, regret and class-block
intervals from supplied scalar records. It does **not** regenerate model outputs
or verify excluded raw tensors. The tiny random DiT example is an offline CPU
plumbing check, not new scientific evidence. No checkpoint is downloaded.

## Scope and limits

Evidence covers DiT-XL/2-256, deterministic 20-step DDIM, stage-local weight-only
fake quantization at W8/W6/W4, and one RTX 5080 platform. Cost means theoretical
packed weight storage—not measured latency, compressed resident VRAM, or kernel
speedup. There is no claim of full allocation, additive joint effects,
activation-quantization support, image-quality improvement, SOTA performance,
or architecture-general validity. FP32 is not ground truth.

The [roadmap](docs/ROADMAP.md) prioritizes a stage-neutral prospective decision
test, then real packed execution with decoded/task quality, before allocation.
External architecture generalization remains open.

Raw tensors, weights, caches, large archives and internal experiment runners are
excluded. Compact evidence and figures have a SHA-256 inventory in
[PUBLIC_EVIDENCE_MANIFEST.json](PUBLIC_EVIDENCE_MANIFEST.json).
Derived P0/P1 artifacts are separate, under
[PUBLIC_AUDIT_MANIFEST.json](PUBLIC_AUDIT_MANIFEST.json); the original scientific
payload and its manifest are unchanged.

## License

Copyright (c) 2026 Munsik Kim. The project's own code, documentation, tables and
figures are licensed under the [MIT License](LICENSE). Retain the copyright
and license notices when copying or redistributing them.

Third-party software and model terms remain separate; see
[attribution](docs/ATTRIBUTION.md). The project license does not remove those
terms.
