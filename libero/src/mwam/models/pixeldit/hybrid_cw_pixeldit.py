from __future__ import annotations


from typing import Any

import torch

import torch.nn as nn

import torch.nn.functional as F


from .modules import get_2d_sincos_pos_embed, precompute_freqs_cis_2d

HYBRID_CW_STREAM_SEMANTICS = "condition_c_and_allxyz_world_w_v1"

HYBRID_CW_ATTENTION_SEMANTICS = "c_from_c_w_from_cw_shared_parameters_v1"

HYBRID_CW_CARRIER_SEMANTICS = "k4_xyz_head_dxyz_wrist_12ch_mosaic_v1"

class HybridCWPixelDiT(nn.Module):
    """K=4 PixelDiT with clean C and heterogeneous Head/Wrist W streams."""

    STREAM_SEMANTICS_VERSION = HYBRID_CW_STREAM_SEMANTICS
    ATTENTION_SEMANTICS_VERSION = HYBRID_CW_ATTENTION_SEMANTICS
    CARRIER_SEMANTICS_VERSION = HYBRID_CW_CARRIER_SEMANTICS


    @staticmethod
    def _layout(height: int, width: int, patch_size: int) -> dict[str, int]:
        if height % 3 or width % 2:
            raise ValueError(
                "Hybrid mosaic must have a 2:1 Head/Wrist height ratio and even width."
            )
        head_height = 2 * (height // 3)
        wrist_height = height // 3
        if any(value % patch_size for value in (height, width, head_height, wrist_height)):
            raise ValueError("Hybrid mosaic regions must all be divisible by patch_size.")
        full_width = width // patch_size
        head_patches = (head_height // patch_size) * full_width
        wrist_patches = (wrist_height // patch_size) * full_width
        return {
            "height": height,
            "width": width,
            "head_height": head_height,
            "wrist_height": wrist_height,
            "patch_height": height // patch_size,
            "patch_width": full_width,
            "head_patches": head_patches,
            "wrist_patches": wrist_patches,
            "patches": head_patches + wrist_patches,
        }

    @staticmethod
    def _patchify(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
        return F.unfold(inputs, kernel_size=patch_size, stride=patch_size).transpose(1, 2)

    @staticmethod
    def _unpatchify(
        patches: torch.Tensor,
        *,
        output_size: tuple[int, int],
        channels: int,
        patch_size: int,
    ) -> torch.Tensor:
        return F.fold(
            patches.transpose(1, 2),
            output_size=output_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def split_flow_carrier(
        self, noised_flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = noised_flow.shape
        if channels != self.flow_channels:
            raise ValueError(
                f"Hybrid carrier requires {self.flow_channels} channels, got {channels}."
            )
        layout = self._layout(height, width, self.patch_size)
        structured = noised_flow.contiguous().view(
            batch_size, self.k, 3, height, width
        )
        w1 = structured[:, :, :, : layout["head_height"], :].contiguous().view(
            batch_size,
            self.head_flow_channels,
            layout["head_height"],
            width,
        )
        w2 = structured[:, :, :, layout["head_height"] :, :].contiguous().view(
            batch_size,
            self.wrist_flow_channels,
            layout["wrist_height"],
            width,
        )
        return w1, w2


    def _fetch_patch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        if key not in self._patch_pos_cache:
            self._patch_pos_cache[key] = precompute_freqs_cis_2d(
                self.head_dim, height, width
            )
        return self._patch_pos_cache[key].to(device=device)

    def _fetch_grouped_pixel_pos(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (height, width)
        if key not in self._pixel_pos_cache:
            position = torch.from_numpy(
                get_2d_sincos_pos_embed(self.pixel_hidden_size, height, width)
            ).view(height, width, self.pixel_hidden_size)
            ph, pw, p = height // self.patch_size, width // self.patch_size, self.patch_size
            position = position.view(ph, p, pw, p, self.pixel_hidden_size)
            position = position.permute(0, 2, 1, 3, 4).contiguous()
            self._pixel_pos_cache[key] = position.view(
                ph * pw, p * p, self.pixel_hidden_size
            )
        return self._pixel_pos_cache[key].to(device=device, dtype=dtype)


    def finish_patch_blocks(self, state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if int(state.get("next_patch_block", -1)) != self.patch_depth:
            raise ValueError("Hybrid Global stack is incomplete.")
        finished_c = F.silu(state["time_embedding_c"] + state["c_patch_tokens"])
        finished_w = F.silu(state["time_embedding_w"] + state["w_patch_tokens"])
        batch_size = int(state["batch_size"])
        state["finished_c_patch_tokens"] = finished_c
        state["finished_w_patch_tokens"] = finished_w
        state["c_patch_condition"] = finished_c.reshape(-1, self.hidden_size)
        state["w_patch_condition"] = finished_w.reshape(-1, self.hidden_size)
        if finished_c.shape[0] != batch_size:
            raise RuntimeError("Hybrid finished patch batch drifted.")
        return state["c_patch_condition"], state["w_patch_condition"]

    def prepare_pixel_tokens(self, state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "c_patch_condition" not in state:
            raise ValueError("Call finish_patch_blocks before Hybrid PiT.")
        layout = state["layout"]
        batch_size = int(state["batch_size"])
        c = self.c_pixel_embedder(
            state["clean_image"],
            img_height=layout["height"],
            img_width=layout["width"],
            patch_size=self.patch_size,
        ).view(batch_size, layout["patches"], self.patch_size**2, self.pixel_hidden_size)
        w1 = self.w1_pixel_embedder(
            state["w1"],
            img_height=layout["head_height"],
            img_width=layout["width"],
            patch_size=self.patch_size,
        ).view(batch_size, layout["head_patches"], self.patch_size**2, self.pixel_hidden_size)
        w2 = self.w2_pixel_embedder(
            state["w2"],
            img_height=layout["wrist_height"],
            img_width=layout["width"],
            patch_size=self.patch_size,
        ).view(batch_size, layout["wrist_patches"], self.patch_size**2, self.pixel_hidden_size)
        w = torch.cat([w1, w2], dim=1)
        if self.use_pixel_abs_pos:
            position = self._fetch_grouped_pixel_pos(
                layout["height"],
                layout["width"],
                device=c.device,
                dtype=c.dtype,
            ).unsqueeze(0)
            c = c + position
            w = w + position
        state["c_pixel_tokens"] = c.view(
            batch_size * layout["patches"], self.patch_size**2, self.pixel_hidden_size
        )
        state["w_pixel_tokens"] = w.view(
            batch_size * layout["patches"], self.patch_size**2, self.pixel_hidden_size
        )
        state["next_pixel_block"] = 0
        return state["c_pixel_tokens"], state["w_pixel_tokens"]

