# Public core and source lineage

Source base: `9f4f253d2d8e284368f73f5de6cead1e622b2063`.
The table identifies source files in the original research snapshot; they are
not dependencies or paths that must exist in this public package.

| Public module | Source basis | Export scope/change |
|---|---|---|
| `quantization.py` | `quant/fake_quant.py` named-module functions | Per-output-channel fake weights; restoration; explicit input/scope guards; no legacy group classifier |
| `residuals.py` | Natural endpoint arithmetic in `e1pred2r2_analysis.py` | Standalone same-state one-step callback; explicit source-native versus FP64 subtraction |
| `geometry.py` | `e1d1.py` ratio-of-sums/contrast contract and central probes | Dependency-free NumPy/Torch diagnostics; no tangent-support claim without validation |
| `controls.py` | `controls/generators.py` | Norm-matched controls; clone donor before scaling |
| `scoring.py` | `e1pred2r2_protocol.py` frozen predictor contract | Immutable per-cell G and beta; no fitting API |
| `decisions.py` | `e1pred2r2_metrics.py` | Exact integer costs, source tie ordering, hybrid orientation, regret and paired class CI; no pandas dependency |
| `verification.py` | Source checksum/tolerance conventions and independent-check design | Streamed hashes, confined inventory paths, separately implemented oracle/regret arithmetic |
| `dit_adapter.py` | `samplers/manual.py` model prediction and eta=0 update | Calls installed DiT/DDIM; explicit crop, class conditioning and stage index; no loader or decoder |

`examples/compact_dit.py` is new synthetic plumbing. `scripts/verify_evidence.py`
recomputes public scalar decisions without importing private analysis runners.
The tests cover core contracts and export-only safety changes, not the whole
historical pipeline. The export has no phase-specific experiment runner.

Excluded are internal CLI/workflow orchestration, corrective/migration code,
model downloads, raw-vector readers tied to private layouts, calibration/refit
runners, precision-bridge studies, failed detector execution, allocation logic,
cloud publishing and previous large export bundles. Their historical outcomes
are preserved, not reclassified.

Evidence/config entries record source commit and source-file SHA in the public
manifest. Original evidence retains source bytes. Added candidate scalar/config
tables are unchanged-value projections from the locked source; removing private
raw references does not create fresh observations. The detailed internal
file-by-file include/exclude review belongs to the owner's release audit, not
the public runtime dependency graph.
