# Attribution and rights review

RCFS-DQ reusable functions and scalar evidence derive from the existing project
at source commit `9f4f253d2d8e284368f73f5de6cead1e622b2063`.
Export adaptations and evidence lineage are described in `CORE_MAP.md` and
`PUBLIC_EVIDENCE_MANIFEST.json`. Packaging is not a new scientific experiment.

Copyright (c) 2026 Munsik Kim. With the owner's authorization, the project's
own code, documentation, scientific tables and figures in this package are
provided under the [MIT License](../LICENSE). This scope includes the
project-owned content in `src/`, `configs/`, `tests/`, `examples/`, `scripts/`,
`docs/`, `evidence/` and `figures/`, as well as the project README and manifest.
Retain the copyright and permission notices when redistributing copies or
substantial portions. Third-party material retains its separate rights and
license terms; this project grant does not relicense it.

The compact example imports installed Hugging Face Diffusers; no Diffusers
implementation is vendored. Diffusers carries an
[Apache-2.0 license](https://github.com/huggingface/diffusers/blob/main/LICENSE).

The historical architecture is DiT, William Peebles and Saining Xie,
*Scalable Diffusion Models with Transformers*. The
[official DiT repository license](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt)
is CC BY-NC 4.0, and the
[DiT-XL-2-256 model card](https://huggingface.co/facebook/DiT-XL-2-256/blob/main/README.md)
also declares CC BY-NC 4.0. A library's license is not a replacement for model
terms. This package includes neither those weights nor a model download step.

NumPy, PyTorch, pytest and packaging dependencies are separately installed and
retain their own licenses. No dataset images or evaluator model is supplied.
This list records attribution considerations, not a legal compatibility opinion.
The project MIT grant does not remove the DiT model's non-commercial terms or
grant rights that belong to third parties. GitHub publication still requires
separate authorization.
