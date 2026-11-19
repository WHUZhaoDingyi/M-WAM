"""Layer-aligned native World -> Action KV; no Action -> World path."""
import torch
from torch import nn
from ..pixeldit_action_mot import PixelDiTActionMoT


class NativeMoT(PixelDiTActionMoT):
    def __init__(self, world, action, cw=False):
        nn.Module.__init__(self)
        if world.bidirectional_world and not cw:
            raise ValueError("Bidirectional World requires CTW KV and joint W/Action sampling")
        self.world_expert, self.action_expert, self.cw = world, action, cw
        self.world_depth, self.pixel_depth, self.action_depth = world.patch_depth, world.pixel_depth, action.num_layers
        self._topology_version = "native_t2i_g14p2_action_mmd14p2_v1"
        self._bridge_semantics_version = ("native_kv_independent_sigma_text_proprio_v2" if action.proprio_in_context
                                          else "native_kv_shared_sigma_condition_text_v1")
        self._interaction_mode = "native_kv"
        self.world_to_action_adapters = nn.ModuleList()
        self.world_native_kv_adapters = nn.ModuleList()
        self.pit_native_kv_adapters = nn.ModuleList()
        self._layer_map = tuple(range(action.num_layers))
        self._attention_runtime_global_step = None
        self.use_gradient_checkpointing = world.use_gradient_checkpointing
        self.text_dim = action.text_dim
        self.mot_dim, self.mot_num_heads = world.hidden_size, world.num_groups
        self.validate_finite = False

    @property
    def world_k_norm(self): return self.world_expert.patch_blocks[0].attn.k_norm_c
    @property
    def action_world_visibility(self): return "c_and_w" if self.cw else "c_only"
    @property
    def bridge_source_schedule(self):
        return tuple(("global_native_kv",i) for i in range(self.world_depth)) + tuple(("pit_native_kv",i) for i in range(self.pixel_depth))
    def _training_kv_gradient_schedule(self): return None

    def forward_world_only(self, world_condition, noised_flow, t_world, world_context, world_context_mask, *, source_support_mask=None):
        return self.world_expert(world_condition,noised_flow,t_world,world_context=world_context,world_context_mask=world_context_mask)

    def forward(self, *, world_condition, noised_flow, t_world, noised_action, t_action,
                context, context_mask, proprio, world_context=None, world_context_mask=None,
                source_support_mask=None, freeze_world=False, return_world_velocity=True):
        if freeze_world: raise ValueError("Native recipe is joint training")
        if self.cw and noised_flow is None: raise ValueError("CW ablation needs a sampled W")
        if not self.action_expert.proprio_in_context and not torch.equal(t_world,t_action):
            raise ValueError("Legacy native recipe requires a shared model timestep")
        world, action = self.world_expert,self.action_expert
        state = world.prepare_tokens(world_condition,noised_flow,t_world,
                    world_context=world_context,world_context_mask=world_context_mask)
        ac = action.prepare_tokens(noised_action,proprio,t_action,context,context_mask)
        for i in range(self.action_depth):
            if i < self.world_depth:
                bank, wbank = world.forward_patch_block(i,state)
                valid = torch.cat((state["text_valid"],state["c_key_valid"]),1)
            else:
                if i == self.world_depth: world.prepare_pixel_tokens(state)
                bank,wbank = world.forward_pixel_block(i-self.world_depth,state)
                valid = state["c_key_valid"]
            if self.cw:
                bank = {key:torch.cat((bank[key],wbank[key]),1) for key in ("k","v")}
                valid = torch.cat((valid,state["c_key_valid"]),1)
            action.forward_block(i,ac,bank,valid)
        return {"world_velocity":world.final_head(state) if return_world_velocity else None,
                "action_velocity":action.final_head(ac)}
