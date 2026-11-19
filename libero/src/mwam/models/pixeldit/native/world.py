"""14 MM-DiT + 2 PiT World, preserving native T2I pixel output mathematics."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..model import PatchTokenEmbedder, PixelTokenEmbedder
from ..modules import TimestepConditioner, FinalLayer
from ..conditioning import WorldConditioningContract
from ..hybrid_cw_pixeldit import HybridCWPixelDiT
from ..t2i_mwam import StableRMSNorm, stable_norms
from .attention import MMBlock, rope_1d
from .pit import DirectedPiT


class NativeWorld(HybridCWPixelDiT):
    def __init__(self, *, views=3, k=4, use_gradient_checkpointing=True,
                 dim=1536, heads=24, pixel_dim=16, pit_dim=1152, pit_heads=16,
                 depth=14, pixel_depth=2, patch_size=16, text_dim=2304, text_length=300,
                 world_attention_mode="directed"):
        nn.Module.__init__(self)
        if world_attention_mode not in ("directed", "bidirectional"):
            raise ValueError("Unknown World attention mode: " + str(world_attention_mode))
        self.world_attention_mode = world_attention_mode
        self.bidirectional_world = world_attention_mode == "bidirectional"
        self.views, self.k, self.patch_size = views, k, patch_size
        self.hidden_size, self.num_groups, self.head_dim = dim, heads, dim // heads
        self.patch_depth, self.pixel_depth, self.pixel_hidden_size = depth, pixel_depth, pixel_dim
        self.clean_channels, self.rgb_channels, self.segmentation_channels = 9, 3, 3
        self.canonical_xyz_channels, self.raw_depth_channels = 3, 0
        self.flow_channels = self.out_channels = self.head_flow_channels = self.wrist_flow_channels = 3 * k
        self.in_channels, self.num_classes = 9 + 3 * k, 0
        self.conditioning_contract = WorldConditioningContract()
        self.use_pixel_abs_pos, self.use_gradient_checkpointing = True, use_gradient_checkpointing
        self.validate_finite = False
        self.text_length, self.pit_dim, self.pit_heads = text_length, pit_dim, pit_heads
        for name, channels in (("c", 9), ("w1", 3*k), ("w2", 3*k)):
            setattr(self, name + "_patch_embedder", PatchTokenEmbedder(channels * patch_size**2, dim))
            setattr(self, name + "_pixel_embedder", PixelTokenEmbedder(channels, pixel_dim, use_pixel_abs_pos=False))
        self.t_embedder = TimestepConditioner(dim)
        self.y_embedder = PatchTokenEmbedder(text_dim, dim, norm_layer=StableRMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.zeros(1, text_length, dim))
        self.stream_embedding = nn.Embedding(2, dim)
        nn.init.normal_(self.stream_embedding.weight, std=.02)
        self.pit_identity_projection = nn.Linear(dim, pit_dim, bias=False)
        self.patch_blocks = nn.ModuleList([MMBlock(dim, heads, True) for _ in range(depth)])
        self.pixel_blocks = nn.ModuleList([DirectedPiT(pixel_dim, dim, pit_dim, pit_heads, patch_size) for _ in range(pixel_depth)])
        # Original FinalLayer: RMSNorm over pixel features -> linear output.
        self.head_final_layer = FinalLayer(pixel_dim, 3*k)
        self.wrist_final_layer = FinalLayer(pixel_dim, 3*k)
        self._patch_pos_cache, self._pixel_pos_cache = {}, {}
        stable_norms(self)

    def _layout(self, h, w, p):
        if self.views == 3:
            return HybridCWPixelDiT._layout(h, w, p)
        if self.views != 2 or h % (2*p) or w % p:
            raise ValueError("Two-view mosaic must split into complete patches")
        half, patches = h // 2, h // (2*p) * (w // p)
        return dict(height=h, width=w, head_height=half, wrist_height=half,
                    patch_height=h//p, patch_width=w//p, head_patches=patches,
                    wrist_patches=patches, patches=2*patches)

    def prepare_tokens(self, clean_image, noised_flow, timestep, *, world_context, world_context_mask, source_support_mask=None):
        if self.bidirectional_world and noised_flow is None:
            raise ValueError("Bidirectional World cannot omit sampled W at inference")
        b, _, h, w = clean_image.shape
        layout = self._layout(h, w, self.patch_size)
        c = self.c_patch_embedder(self._patchify(clean_image, self.patch_size))
        wp = w1 = w2 = None
        if noised_flow is not None:
            w1, w2 = self.split_flow_carrier(noised_flow)
            wp = torch.cat((self.w1_patch_embedder(self._patchify(w1, self.patch_size)),
                            self.w2_patch_embedder(self._patchify(w2, self.patch_size))), 1)
        time = self.t_embedder(timestep.float()).reshape(b, 1, self.hidden_size)
        length = min(world_context.shape[1], self.text_length)
        y = self.y_embedder(world_context[:, :length])
        y = y + self.y_pos_embedding[:, :length].to(y.dtype)
        identities = self.stream_embedding(torch.arange(2, device=c.device))
        return dict(c_patch_tokens=c, w_patch_tokens=wp, text_tokens=y,
            text_valid=world_context_mask[:, :length].bool(), text_pos=rope_1d(length, self.head_dim, c.device),
            condition=F.silu(time), time_embedding_c=time, time_embedding_w=time,
            position=self._fetch_patch_pos(layout["patch_height"], layout["patch_width"], c.device),
            identities=identities, pit_identities=self.pit_identity_projection(identities),
            clean_image=clean_image, noised_flow=noised_flow, w1=w1, w2=w2,
            batch_size=b, layout=layout, next_patch_block=0,
            c_key_valid=torch.ones(c.shape[:2], device=c.device, dtype=torch.bool))

    def forward_patch_block(self, i, state, *, return_native_kv=True):
        block = self.patch_blocks[i]
        args = [state[k] for k in ("c_patch_tokens", "w_patch_tokens", "text_tokens",
            "condition", "position", "text_pos", "text_valid", "identities")]
        args.append(self.bidirectional_world)
        result = checkpoint(block.world, *args, use_reentrant=False) if self.training and self.use_gradient_checkpointing else block.world(*args)
        state["c_patch_tokens"], state["w_patch_tokens"], state["text_tokens"] = result[:3]
        state["next_patch_block"] = i+1
        # World text/C keys are captured at this layer's native attention input.
        bank = {"k": torch.cat((result[5], result[3]), 1), "v": torch.cat((result[6], result[4]), 1)}
        return bank, {"k": result[7], "v": result[8]}

    def prepare_pixel_tokens(self, state):
        if state["w_patch_tokens"] is not None:
            self.finish_patch_blocks(state)
            return super().prepare_pixel_tokens(state)
        layout, b = state["layout"], state["batch_size"]
        state["c_patch_condition"] = F.silu(state["time_embedding_c"] + state["c_patch_tokens"]).reshape(-1, self.hidden_size)
        state["w_patch_condition"] = None
        pixels = self.c_pixel_embedder(state["clean_image"], img_height=layout["height"],
            img_width=layout["width"], patch_size=self.patch_size)
        grouped = pixels.reshape(b, layout["patches"], self.patch_size**2, self.pixel_hidden_size)
        grouped = grouped + self._fetch_grouped_pixel_pos(layout["height"], layout["width"],
            device=pixels.device, dtype=pixels.dtype)[None]
        state["c_pixel_tokens"] = grouped.reshape_as(pixels)
        state["w_pixel_tokens"], state["next_pixel_block"] = None, 0

    def forward_pixel_block(self, i, state, *, return_native_kv=True):
        block, layout = self.pixel_blocks[i], state["layout"]
        pos = block.condition._fetch_pos(layout["patch_height"], layout["patch_width"], state["clean_image"].device)
        args = [state[k] for k in ("c_pixel_tokens", "w_pixel_tokens", "c_patch_condition", "w_patch_condition")]
        args += [state["batch_size"], pos, state["pit_identities"], self.bidirectional_world]
        result = checkpoint(block, *args, use_reentrant=False) if self.training and self.use_gradient_checkpointing else block(*args)
        state["c_pixel_tokens"], state["w_pixel_tokens"] = result[:2]
        state["next_pixel_block"] = i+1
        return {"k":result[2], "v":result[3]}, {"k":result[4], "v":result[5]}

    def final_head(self, state):
        layout, b, p = state["layout"], state["batch_size"], self.patch_size
        pixels = state["w_pixel_tokens"].reshape(b, layout["patches"], p*p, self.pixel_hidden_size)
        head, wrist = pixels.split((layout["head_patches"], layout["wrist_patches"]), 1)
        outputs = []
        for tokens, layer, height in ((head, self.head_final_layer, layout["head_height"]),
                                     (wrist, self.wrist_final_layer, layout["wrist_height"])):
            out = layer(tokens).permute(0,1,3,2).reshape(b, tokens.shape[1], self.flow_channels*p*p)
            outputs.append(self._unpatchify(out, output_size=(height, layout["width"]),
                                           channels=self.flow_channels, patch_size=p))
        return torch.cat(outputs, 2)

    def forward(self, clean_image, noised_flow, timestep, *, world_context, world_context_mask, source_support_mask=None):
        state = self.prepare_tokens(clean_image, noised_flow, timestep, world_context=world_context, world_context_mask=world_context_mask)
        for i in range(self.patch_depth): self.forward_patch_block(i, state)
        self.prepare_pixel_tokens(state)
        for i in range(self.pixel_depth): self.forward_pixel_block(i, state)
        return self.final_head(state)
