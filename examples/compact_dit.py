"""Offline CPU plumbing smoke with a tiny random DiT; not research evidence."""

import json
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DiTTransformer2DModel
from torch.nn.attention import SDPBackend, sdpa_kernel

from rcfs_dq.controls import isotropic, seed_shuffled
from rcfs_dq.dit_adapter import DDIMPath
from rcfs_dq.geometry import finite_horizon_response, lcg_ratio_of_sums, log_contrasts
from rcfs_dq.residuals import extract_stage_residual, squared_energy


def smoke():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/compact_dit.json").read_text()
    )
    torch.manual_seed(config["synthetic_seed"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = DiTTransformer2DModel(**config["transformer"]).eval()
    calls = []

    def count_forward(_model, _args, _output):
        calls.append(1)

    counter = model.register_forward_hook(count_forward)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    path = DDIMPath(model, scheduler, [0], num_steps=config["steps"])
    state = torch.randn(1, 4, 8, 8)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        anchor = path.update(state, 0)

        def step(x):
            return path.update(x, 1)

        obs = extract_stage_residual(model, step, anchor, ["transformer_blocks.0.ff.net.2"], 4)
        assert obs.restored and torch.equal(obs.baseline, step(anchor))
        directions = {
            "actual": obs.residual.float(),
            "isotropic": isotropic(obs.residual.float(), 7),
            "shuffled": seed_shuffled(
                obs.residual.float(), torch.roll(obs.residual.float(), 1, -1)
            ),
        }
        gains = {}
        for name, direction in directions.items():
            result = finite_horizon_response(
                lambda x: path.continue_from(x, 2), obs.baseline, direction, config["alpha"]
            )
            gains[name] = lcg_ratio_of_sums(
                [float(torch.linalg.vector_norm(result["derivative"]))],
                [float(torch.linalg.vector_norm(direction.double()))],
            )
        counter.remove()
        assert len(calls) == 16
        return dict(
            passed=True,
            kind="synthetic random tiny DiT CPU smoke",
            pretrained_model_used=False,
            weight_restoration=True,
            residual_energy=float(squared_energy(obs.residual)[0]),
            gains=gains,
            contrasts=log_contrasts(gains["actual"], gains["shuffled"], [gains["isotropic"]]),
            model_forwards=len(calls),
            scientific_claim=False,
        )


if __name__ == "__main__":
    print(json.dumps(smoke(), indent=2, allow_nan=False))
