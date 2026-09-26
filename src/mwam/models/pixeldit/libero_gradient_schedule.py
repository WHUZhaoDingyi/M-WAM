"""Forward-identical, backward-only Action -> World KV scheduling."""
from __future__ import annotations

import torch

SCHEMA = "libero_action_to_world_gradient_v1"
PLUS_SCHEMA = "libero_plus_action_to_world_gradient_50k_v1"
PLUS_SCHEDULE = dict(schema_version=PLUS_SCHEMA, detach_steps=20000,
                     ramp_steps=1000, initial_scale=0.5, final_scale=1.0,
                     stage_steps=50000, source_step=0)


def validate_schedule(value):
    value = dict(value)
    expected = dict(schema_version=SCHEMA, detach_steps=20000,
                    ramp_steps=333, initial_scale=0.5, final_scale=1.0,
                    stage_steps=30000, source_step=30000)
    if value.get('schema_version') == PLUS_SCHEMA:
        expected = PLUS_SCHEDULE
    elif value.get('schema_version') == 'libero_action_to_world_gradient_50k_v1':
        expected = dict(expected, schema_version='libero_action_to_world_gradient_50k_v1',
                        ramp_steps=1000, stage_steps=50000)
    if value != expected:
        raise ValueError("LIBERO gradient schedule must match the approved 20K/333/30K contract")
    return value


def gradient_scale(completed_steps, schedule):
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("Gradient schedule requires completed successful stage steps")
    if completed_steps < schedule["detach_steps"]:
        return 0.0
    fraction = min((completed_steps - schedule["detach_steps"]) / schedule["ramp_steps"], 1.0)
    return schedule["initial_scale"] + fraction * (schedule["final_scale"] - schedule["initial_scale"])


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.scale, None


def scale_gradient(value, scale):
    """Keep the autograd topology fixed even at zero; no forward arithmetic."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("KV gradient scale must be in [0,1]")
    return _ScaleGradient.apply(value, float(scale))
