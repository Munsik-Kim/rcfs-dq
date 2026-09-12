"""Explicit DiT model/scheduler boundary; no checkpoint loader or downloads."""

import torch


class DDIMPath:
    def __init__(self, transformer, scheduler, class_ids, num_steps=20):
        if scheduler.__class__.__name__ != "DDIMScheduler":
            raise ValueError("Only the documented DDIM scheduler is supported")
        self.model, self.scheduler = transformer, scheduler
        self.device = next(transformer.parameters()).device
        self.class_ids = tuple(class_ids)
        scheduler.set_timesteps(num_steps, device=self.device)
        self.timesteps = scheduler.timesteps.detach().clone()

    @torch.inference_mode()
    def update(self, state, stage):
        if state.shape[0] != len(self.class_ids) or not 0 <= stage < len(self.timesteps):
            raise ValueError("Conditioning or stage mismatch")
        t = self.timesteps[stage]
        x = self.scheduler.scale_model_input(state, t)
        labels = torch.as_tensor(self.class_ids, device=self.device, dtype=torch.long)
        prediction = self.model(
            x, timestep=t.reshape(1).expand(state.shape[0]), class_labels=labels
        ).sample
        channels = self.model.config.in_channels
        if self.model.config.out_channels // 2 == channels:
            prediction = prediction[:, :channels]
        return self.scheduler.step(prediction, t, state, eta=0.0).prev_sample

    @torch.inference_mode()
    def continue_from(self, state, start_stage):
        if not 0 <= start_stage <= len(self.timesteps):
            raise ValueError("Invalid suffix boundary")
        x = state.clone()
        for stage in range(start_stage, len(self.timesteps)):
            x = self.update(x, stage)
        return x
