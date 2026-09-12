# Changelog

## 0.1.1

- Remove internal publication notes from public documentation and the unrelated
  `authorized_to_push` key from evidence-verifier JSON. Scientific values,
  comparison counts, verification scope and `model_calls` are unchanged.
- Fix norm correction for noncontiguous controls; validate shape, dtype and
  finiteness without modifying caller-owned inputs.
- Reject unsupported quantizer ranges and handle extreme LCG ratios explicitly.
- Bound Python support to 3.11–3.12; document CPU, development and offline
  installation and add pinned CPU continuous-integration checks.

Evidence, frozen configurations, figures and the MIT license are unchanged.
