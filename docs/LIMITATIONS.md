# Limitations

- J0/V3/BRIDGE0 cover one DiT-XL/2-256, deterministic DDIM20, CFG off, eta=0,
  and an exact-promoted FP32 path. Scheduler-only FP64 is not an FP64 model.
- W4 on selected all-block QKV, attention-output and MLP-down groups, eight
  classes and four seeds do not establish external architecture/quantizer or
  population-wide generalization.
- V3's +1/128 target differs from BRIDGE0's full natural error. Its half-alpha
  target also participates in qualification; replay is not independent
  confirmation. C/D failures remain visible.
- BRIDGE0 is retrospective. Its estimator was locked before natural-error
  computation, after V3 had been seen—not fresh prospective preregistration.
- Ninety-six probe16→17 targets are missing. No imputation, three-probe aggregate
  or prediction about missing outcomes is allowed.
- Full uses two additional directional predictions per direction. Baseline and
  quantized trajectories/residuals also cost compute. This is not equal-compute
  deployment improvement or a free predictor.
- Natural errors follow one stage4 intervention with restored weights, not
  persistent quantization. Next-state response error is not terminal risk,
  quantizer-selection regret, decoded/task quality or image quality.
- J0/V3/BRIDGE0 establish no speed, packed-storage, VRAM, allocation or safety
  improvement. Public scalar checks do not regenerate hidden raw model outputs.
- A confidence interval crossing zero establishes neither superiority nor
  equivalence. Frequent wins do not neutralize heavy tails under a mean endpoint.
  No post-hoc QKV/attention-versus-MLP routing policy is validated.
- Scalar/Jacobian conclusions are scope-specific: no invalidation of TCEC,
  RTN/GPTQ comparison, uniquely larger actual-residual omission, or universal
  uselessness of Jacobian methods is established.

See [results](RESULTS.md) for unfavorable controls and tails alongside surviving
findings. Historical scientific decisions remain unchanged.
