# RCFS-DQ

Residual-conditioned finite-horizon sensitivity for diffusion quantization.

**Actual quantization residual → future-sensitive geometry → prospective
matched-cost decision utility.**

RCFS-DQ asks whether actual quantization residuals occupy future-sensitive
directions non-randomly, and whether that structure improves quantization
decisions. It does not claim that finite-time sensitivity itself is new.

## Evidence in brief

- Core81 discovery: 81/81 cells met the locked tangent-support criterion.
- Independent confirmation: positive-family mean 1.9092, one-sided exact
  p = 0.00390625; expected sign in 9/9 cells, stricter replication in 8/9.
- Prospective native-FP16 decisions: 16 fresh classes × 4 seeds, 27 exactly
  matched-cost stage-selection sets. Frozen Candidate-B/Hybrid mean normalized
  regret was 0.0431, versus Energy 0.1852 and source Constant 0.0460.
- The strong source Constant is competitive: strict normalized-regret
  superiority over it was **not established**.
- Negative result: the simple one-step exception detector failed
  (recall 3.80%, FPR 28.74%, AP 0.0519). E1-Tail0 remains a negative result.

See [results](docs/RESULTS.md) for intervals and limitations and
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

External architecture generalization and real packed-path utility remain open.
The [roadmap](docs/ROADMAP.md) puts those checks before allocation.

Raw tensors, weights, caches, large archives and internal experiment runners are
excluded. Compact evidence and figures have a SHA-256 inventory in
[PUBLIC_EVIDENCE_MANIFEST.json](PUBLIC_EVIDENCE_MANIFEST.json).

## License

Copyright (c) 2026 Munsik Kim. The project's own code, documentation, tables and
figures are licensed under the [MIT License](LICENSE). Retain the copyright
and license notices when copying or redistributing them.

Third-party software and model terms remain separate; see
[attribution](docs/ATTRIBUTION.md). The project license does not remove those
terms.
