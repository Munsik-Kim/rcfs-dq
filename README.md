# RCFS-DQ

RCFS-DQ studies whether actual weight-quantization residuals interact with
future-sensitive diffusion geometry, and whether measuring that information
helps prediction or decisions beyond strong simple baselines. It provides
measurement primitives and auditable evidence—not a deployment optimizer.

**Actual residual → geometry and strong-baseline audit → scoped local prediction
→ natural-error limitation.**

## What survived / what did not

| Question | Evidence-supported result |
|---|---|
| Does geometry improve on Energy? | Yes, in the original selected prospective matched-cost task. |
| Additional adaptive utility over strong static/stage baselines? | Not established; latest-stage had lower mean absolute regret. |
| Local directional/Jacobian information measurable and predictive? | Yes, within the locked small-radius DiT/W4 V3 scope; Full improved over No-J and frozen Scalar. |
| Stable mean benefit over Scalar at natural error scale? | Not established: BRIDGE0's primary interval crosses zero despite 178/192 wins. |
| Actual residual uniquely more omission-sensitive than controls? | Not supported: observed actual R_J was lower than donor/isotropic. |
| Validated deployment, quality, memory or speed improvement? | No. |

Current project-level status: **RCFS_DQ_LOCAL_MECHANISM_TRACK_CLOSED_AS_LIMITED**.
This closes a research scope; it does not replace historical decisions. V1's
numerical failure and the failed E1-Tail0 exception detector remain negative
results. Local predictive information does not establish stable natural-scale
utility. BRIDGE0 covers only probe5→6 and probe11→12; 96 probe16→17 targets are
missing, with no three-probe aggregate. FP32 is not ground truth.

P0/P1 and BRIDGE0 are retrospective CPU-only audits, not fresh confirmation.
For those audits: model calls = 0, GPU calls = 0, new model observations = 0,
predictor refits = 0, new prospective confirmation = 0. V3 is a separately
recorded historical model measurement; this publication adds no experiment.

Read the [research summary](docs/RESEARCH_SUMMARY.md),
[results and intervals](docs/RESULTS.md), [limitations](docs/LIMITATIONS.md),
and [reproducibility boundaries](docs/REPRODUCIBILITY.md).
The [roadmap](docs/ROADMAP.md) prioritizes adaptive-headroom audits and
ModelDiff Guard / quant-qualify; it does not authorize a mechanism-rescue run.

## Use and verify

Python 3.11–3.12; use the [documented CPU constraints](requirements/cpu-constraints.txt).
From a repository checkout with those dependencies already installed:

```bash
python scripts/verify_evidence.py
python -O scripts/verify_evidence.py
python scripts/verify_public_audits.py
python scripts/verify_closeout_evidence.py
python -O scripts/verify_closeout_evidence.py
python -m pytest -q
```

See [installation and offline wheel checks](docs/REPRODUCIBILITY.md),
[reusable core](docs/CORE_MAP.md), and the optional
[tiny random CPU DiT example](examples/compact_dit.py). Synthetic software tests
are not scientific observations. Public scalar verification cannot regenerate
excluded raw tensors or historical model outputs.

Provenance layers remain separate:
[original evidence](PUBLIC_EVIDENCE_MANIFEST.json),
[P0/P1 derived audit](PUBLIC_AUDIT_MANIFEST.json), and
[local/natural closeout evidence](PUBLIC_CLOSEOUT_MANIFEST.json).
Weights, raw tensor containers, caches and review ZIPs are excluded.

## License

Copyright (c) 2026 Munsik Kim. Project code, documentation, tables and figures
use the [MIT License](LICENSE); retain its notices when redistributing.
[Third-party software and model terms](docs/ATTRIBUTION.md) remain separate.
