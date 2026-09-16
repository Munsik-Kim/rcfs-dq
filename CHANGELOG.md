# Changelog

## Unreleased

### Dependency security

- Separate project input/build/install gates from direct upstream JIT sentinels;
  retain the related native-crash observations without publishing their payloads.
- Record the supported API reachability boundary and official-advisory distinction.
  This does not assert that all PyTorch JIT inputs are safe or change any science.
- Raise the current public runtime to torch >=2.14,<2.15 (CPU CI: 2.14.0+cpu)
  and both runtime and isolated-build setuptools requirements to >=83,<84
  (CI: 83.0.0). Include the build frontend in the test extra.
- Exclude the dependency ranges flagged by GHSA-rrmf-rvhw-rf47,
  GHSA-5rjg-fvgr-3xxf and GHSA-h35f-9h28-mq5c. Historical environments and
  scientific artifacts are not upgraded or relabeled.
- Add installed-version and fixed-boundary regression checks. Package version
  remains 0.1.1; this entry does not create a release or a new scientific result.

### Limited local-mechanism closeout

- Preserve the qualified V3 small-radius result separately from retrospective
  BRIDGE0 natural-error limits, including 178/192 wins, mean uncertainty,
  MLP-down tails and 96 missing probe16 targets.
- Close the local mechanism-to-natural-utility scope as limited, without
  replacing historical scientific decisions or claiming deployment utility.
- Add byte-preserved scalar/decision evidence, a separate closeout manifest and
  independent model-free public arithmetic/CI verification.
- Replace the former next-step priority below with adaptive-headroom research
  and ModelDiff Guard / quant-qualify. J1, P2 and probe16 supplement are deferred.
- No package version bump, tag, release, new scientific run or predictor refit.

### Earlier P0/P1 public integration

- Add a separate, hash-inventoried retrospective P0/P1 scalar audit layer:
  free/static policies, conditional headroom and aligned geometry ablations.
- Preserve Energy improvement while narrowing adaptive-utility and pairing
  claims in light of competitive Constant/stage priors and absolute-risk tails.
- Prioritize a future stage-neutral test of absolute selection regret; retain
  E1-Tail0 as a negative result. No new model observations or predictor fitting.
- Add independent audit checks and verification that remains active under
  optimized Python, plus repository security policy and automation.

Original scientific evidence, frozen configurations, figures, their manifest
and the MIT license remain unchanged. This is not a new package release.

## 0.1.1

- Remove internal publication notes from public documentation and the unrelated
  publication-permission key from evidence-verifier JSON. Scientific values,
  comparison counts, verification scope and `model_calls` are unchanged.
- Fix norm correction for noncontiguous controls; validate shape, dtype and
  finiteness without modifying caller-owned inputs.
- Reject unsupported quantizer ranges and handle extreme LCG ratios explicitly.
- Bound Python support to 3.11–3.12; document CPU, development and offline
  installation and add pinned CPU continuous-integration checks.

Evidence, frozen configurations, figures and the MIT license are unchanged.
