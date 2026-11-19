"""Construct the two-view LIBERO World and Action branches."""
import torch

from ...wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
    WanHighNoise405505UniformWeightFlowMatchScheduler,
    WanUniformSigmaUniformWeightFlowMatchScheduler,
)
from ..t2i_libero import LiberoWorldConditioning
from .action import NativeAction
from .initialization import load_official
from .interaction import NativeMoT
from .policy import NativePolicy
from .world import NativeWorld


def build(*, geometry, kflow, action_norm, proprio_norm, pixeldit_pretrained_path,
          device, use_gradient_checkpointing, uniform_noise=False):
    scheduler = (WanUniformSigmaUniformWeightFlowMatchScheduler if uniform_noise
                 else WanHighNoise405505UniformWeightFlowMatchScheduler)
    train_world, train_action = scheduler(shift=5.), scheduler(shift=5.)
    infer_world, infer_action = WanContinuousFlowMatchScheduler(), WanContinuousFlowMatchScheduler()
    with torch.device(device):
        world = NativeWorld(
            views=2, k=kflow.k, text_length=301, world_attention_mode="directed",
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        world.conditioning_contract = LiberoWorldConditioning()
        action = NativeAction(
            action_dim=action_norm.feature_dim, proprio_dim=proprio_norm.feature_dim,
            text_length=301, timewise=False, proprio_in_context=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        model = NativePolicy(
            NativeMoT(world, action, cw=False), kflow_contract=kflow,
            normalization_contract=geometry, action_horizon=32,
            action_normalization_contract=action_norm, proprio_dim=proprio_norm.feature_dim,
            proprio_normalization_contract=proprio_norm, text_dim=2304,
            timestep_coupling="independent", train_world_scheduler=train_world,
            train_action_scheduler=train_action, infer_world_scheduler=infer_world,
            infer_action_scheduler=infer_action, loss_lambda_momap=1.,
            loss_lambda_action=1., method_contract=dict(
                schema_version="mwam_native_t2i_method_v1", train_timestep_coupling="independent",
                model_time="1000*sigma", world_attention="C/Y cannot read W",
                action_world_visibility="C_text", action_tokenization="trajectory32_context_proprio",
                source_identity="embedding_after_rotated_QK_every_layer",
            ),
        )
    model.pretrained_load_reports = {"t2i": load_official(world, action, pixeldit_pretrained_path)}
    return model
