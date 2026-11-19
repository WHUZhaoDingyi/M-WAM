"""Trajectory-plus-current-state tokens and a 14+2 MM-DiT Action branch."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..model import PatchTokenEmbedder
from ..modules import TimestepConditioner
from ..pixeldit_aligned_action import fixed_1d_sincos_position_embedding
from ..t2i_mwam import StableRMSNorm
from .attention import MMBlock, rope_1d


class NativeAction(nn.Module):
    def __init__(self, action_dim=14, proprio_dim=14, horizon=32, dim=1536,
                 heads=24, tail_dim=1152, tail_heads=16, depth=14, tail_depth=2,
                 text_dim=2304, text_length=300, timewise=False, use_gradient_checkpointing=True,
                 proprio_in_context=False):
        super().__init__()
        if (action_dim, proprio_dim) not in ((14,14),(7,8)):
            raise ValueError("Explicit RobotWin 14/14 or LIBERO 7/8 mapping required")
        self.proprio_in_context = bool(proprio_in_context)
        self.action_dim, self.proprio_dim, self.action_horizon = action_dim, proprio_dim, horizon
        self.hidden_size, self.tail_dim, self.num_heads, self.tail_heads = dim, tail_dim, heads, tail_heads
        self.num_layers, self.global_depth, self.text_dim = depth+tail_depth, depth, text_dim
        self.timewise, self.use_gradient_checkpointing = timewise, use_gradient_checkpointing
        if timewise and action_dim != proprio_dim and not proprio_in_context:
            raise ValueError("Time-wise state token requires aligned action/proprio dimensions")
        self.action_encoder = nn.Linear(action_dim if timewise else horizon+(not proprio_in_context), dim)
        self.gripper_encoder = nn.Linear(horizon+2, dim) if action_dim==7 and not timewise and not proprio_in_context else None
        self.y_embedder = PatchTokenEmbedder(text_dim, dim, norm_layer=StableRMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.zeros(1, text_length, dim))
        self.t_embedder = TimestepConditioner(dim)
        self.width_bridge = nn.Conv1d(dim, tail_dim, 1)
        self.time_bridge = nn.Linear(dim, tail_dim)
        self.blocks = nn.ModuleList([MMBlock(dim,heads) for _ in range(depth)] +
                                   [MMBlock(tail_dim,tail_heads) for _ in range(tail_depth)])
        self.action_decoder = nn.Linear(tail_dim, action_dim if timewise else horizon)
        self.register_buffer("joint_pos_embed", fixed_1d_sincos_position_embedding(horizon+(not proprio_in_context) if timewise else action_dim, dim), persistent=False)
        nn.init.xavier_uniform_(self.action_encoder.weight)
        nn.init.zeros_(self.action_encoder.bias)
        nn.init.zeros_(self.action_decoder.weight)
        nn.init.zeros_(self.action_decoder.bias)

    def tokenize(self, action, proprio):
        b = action.shape[0]
        if action.shape[1:] != (self.action_horizon, self.action_dim) or proprio.shape != (b,1,self.proprio_dim):
            raise ValueError("Action/proprio shapes differ from native task contract")
        if self.proprio_in_context:
            x = self.action_encoder(action if self.timewise else action.transpose(1,2))
        elif self.timewise:
            x = self.action_encoder(torch.cat((action, proprio), dim=1))
        else:
            trajectories, state = action.transpose(1,2), proprio.transpose(1,2)
            if self.gripper_encoder is None:
                x = self.action_encoder(torch.cat((trajectories,state), -1))
            else:
                arm = self.action_encoder(torch.cat((trajectories[:,:6],state[:,:6]), -1))
                # Preserve BOTH gripper qpos values; neither is dropped or averaged.
                gripper = self.gripper_encoder(torch.cat((trajectories[:,6:7],proprio[:,:,6:8]), -1))
                x = torch.cat((arm,gripper), 1)
        return x + self.joint_pos_embed.to(x)

    def prepare_tokens(self, action, proprio, timestep, context, context_mask):
        x = self.tokenize(action,proprio)
        if self.proprio_in_context and context.shape[1] > self.y_pos_embedding.shape[1]:
            raise ValueError("Text/proprio context would be truncated")
        length = min(context.shape[1], self.y_pos_embedding.shape[1])
        y = self.y_embedder(context[:,:length]) + self.y_pos_embedding[:,:length].to(x.dtype)
        time = self.t_embedder(timestep.float()).reshape(x.shape[0],1,self.hidden_size)
        return dict(tokens=x,text=y,text_valid=context_mask[:,:length].bool(),
                    condition=F.silu(time),tail_condition=F.silu(self.time_bridge(time)))

    def forward_block(self, i, state, bank, valid):
        if i == self.global_depth:
            # The same 1x1 projection adapts both residual token streams.
            for key in ("tokens","text"):
                state[key] = self.width_bridge(state[key].transpose(1,2)).transpose(1,2)
            state["condition"] = state["tail_condition"]
        block = self.blocks[i]
        dim = block.attn.head_dim
        pos = rope_1d(state["text"].shape[1], dim, state["tokens"].device)
        args = (state["tokens"],state["text"],state["condition"],pos,state["text_valid"],
                bank["k"],bank["v"],valid)
        out = checkpoint(block.action,*args,use_reentrant=False) if self.training and self.use_gradient_checkpointing else block.action(*args)
        state["tokens"],state["text"] = out

    def final_head(self, state):
        out = self.action_decoder(state["tokens"])
        return out[:, :self.action_horizon] if self.timewise else out.transpose(1,2)
