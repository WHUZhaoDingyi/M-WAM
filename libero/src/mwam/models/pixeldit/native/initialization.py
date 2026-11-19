"""Auditable native T2I mapping; no layer duplication and no silent partial loads."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from ..t2i_mwam import T2I_SHA256


def load_official(world, action, path):
    source = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for key in ("model","state_dict","ema"):
        if key in source and isinstance(source[key],dict):
            source = source[key]
            break
    source = {k.removeprefix("core."):v for k,v in source.items()}
    report = {"source":str(Path(path).resolve()),"expected_sha256":T2I_SHA256,
              "global_source_layers":list(range(world.patch_depth)),
              "pixel_source_layers":list(range(world.pixel_depth)),
              "exact":{}, "adapted":{}, "new":[]}
    # Native checkpoint dimensions and block counts are required, not guessed.
    if {int(k.split(".")[1]) for k in source if k.startswith("patch_blocks.")} != set(range(world.patch_depth)):
        raise ValueError("Checkpoint MM-DiT depth differs from native architecture")
    if {int(k.split(".")[1]) for k in source if k.startswith("pixel_blocks.")} != set(range(world.pixel_depth)):
        raise ValueError("Checkpoint PiT depth differs from native architecture")

    def copy_tensor(target, src, name, mode=None):
        if src not in source:
            raise KeyError("Missing pretrained parameter: "+src)
        value = source[src]
        if mode is None:
            if target.shape != value.shape:
                # Revised LIBERO appends one proprio token to the official
                # 300-token text context. Only that exact extension is allowed;
                # the new position keeps its independently initialized value.
                if (src == "y_pos_embedding" and value.ndim == target.ndim == 3
                        and target.shape[:2] == (1, 301)
                        and value.shape == (1, 300, target.shape[2])):
                    target[:, :300].copy_(value)
                    report["adapted"][name] = src + " -> first 300 text positions"
                    report["new"].append(name + "[:,300:]")
                    return
                raise ValueError(f"Pretrained shape mismatch: {src} -> {name}")
            target.copy_(value); report["exact"][name] = src
        else:
            target.copy_(mode(value).to(target)); report["adapted"][name] = src

    with torch.no_grad():
        for key, target in world.state_dict().items():
            name = "world."+key
            if key.startswith("patch_blocks."):
                src = key.replace("norm_c","norm_x").replace("mlp_c","mlp_x").replace("modulation_cond","modulation_img")
                for part in ("qkv","q_norm","k_norm","proj"):
                    src = src.replace("attn."+part+"_c","attn."+part+"_x")
                copy_tensor(target,src,name)
            elif key.startswith("pixel_blocks."):
                src = key.replace(".condition.",".").replace(".target.",".")
                copy_tensor(target,src,name)
            elif key.startswith(("t_embedder.","y_embedder.")) or key=="y_pos_embedding":
                copy_tensor(target,key,name)
            elif key.startswith(("c_patch_embedder.","c_pixel_embedder.")):
                src = key.replace("c_patch_embedder","s_embedder").replace("c_pixel_embedder","pixel_embedder")
                if key.endswith("proj.weight"):
                    value = source[src]; target.zero_(); target[:,:value.shape[1]].copy_(value)
                    report["adapted"][name] = src+" -> RGB slice; other source channels zero"
                else: copy_tensor(target,src,name)
            elif key.startswith(("w1_patch_embedder.","w2_patch_embedder.","w1_pixel_embedder.","w2_pixel_embedder.")):
                src = key.replace("w1_patch_embedder","s_embedder").replace("w2_patch_embedder","s_embedder").replace("w1_pixel_embedder","pixel_embedder").replace("w2_pixel_embedder","pixel_embedder")
                if key.endswith("proj.weight"):
                    copy_tensor(target,src,name,lambda v:v.repeat(1,world.k)/math.sqrt(world.k))
                    report["adapted"][name] += " -> repeat across horizons / sqrt(K)"
                else: copy_tensor(target,src,name)
            elif key.startswith(("head_final_layer.","wrist_final_layer.")):
                src = key.replace("head_final_layer","final_layer").replace("wrist_final_layer","final_layer")
                if ".linear." in key:
                    copy_tensor(target,src,name,lambda v:v.repeat(world.k,1) if v.ndim==2 else v.repeat(world.k))
                    report["adapted"][name] += " -> repeated XYZ output per horizon"
                else: copy_tensor(target,src,name)
            else: report["new"].append(name)
        for key,target in action.state_dict().items():
            name="action."+key
            if key.startswith("blocks.") and int(key.split(".")[1]) < action.global_depth:
                copy_tensor(target,key.replace("blocks.","patch_blocks.",1),name)
            elif key.startswith(("t_embedder.","y_embedder.")) or key=="y_pos_embedding":
                copy_tensor(target,key,name)
            else: report["new"].append(name)
    report["new_policy"] = "New IO/width/time/identity adapters; Action tail two complete MM-DiT blocks. Native matched backbone tensors loaded exactly."
    return report

