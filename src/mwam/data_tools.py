"""Prepare training inputs from RoboTwin episode exports."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare_data(manifest, template, text_cache, seg_manifest, output):
    from .robotwin_momap.raw_episode_hdf5 import RawEpisodeReader
    from .datasets.robotwin_momap.prompt_index import (
        RouteBPromptIndexResolver, task_info_key, PROMPT_INDEX_SCHEMA_VERSION,
    )
    from .datasets.robotwin_momap.v1_source import DEFAULT_PROMPT
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new prepared-data directory: {output}")
    rows = json.loads(manifest.read_text())["episodes"]
    if not rows:
        raise ValueError("episodes must be nonempty")
    episodes, records, perception_rows = [], {}, []
    for row in rows:
        path = Path(row["episode_path"]).expanduser()
        path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if str(path) in episodes:
            raise ValueError(f"Duplicate episode: {path}")
        reader = RawEpisodeReader(path)
        try:
            metadata = reader.metadata
            frame_count = reader.num_frames
        finally:
            reader.close()
        instruction = row["instruction"]
        if not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip():
            raise ValueError(f"Invalid instruction for {path}")
        key = (metadata["task_name"], metadata["task_config"],
               task_info_key(RouteBPromptIndexResolver._current_task_info(metadata)))
        if key in records and records[key]["instruction"] != instruction:
            raise ValueError(f"Conflicting instructions for scene identity {key}")
        records[key] = dict(task_name=key[0], task_config=key[1], task_info_key=key[2], instruction=instruction)
        episodes.append(str(path))
        if frame_count < 2:
            raise ValueError(f"Episode has fewer than two frames: {path}")
        perception_rows.append(dict(episode_id=str(metadata.get("rollout_id") or path.stem),
            episode_path=str(path), task_name=key[0], variant=key[1], instruction=instruction,
            expected_frame_count=frame_count, source=dict(legal_source_index_start=1,
                legal_source_index_stop_inclusive=frame_count-1, legal_source_count=frame_count-1,
                causal_processing_frame_count=frame_count)))
    supplied_seg = seg_manifest is not None
    seg_manifest = Path(seg_manifest).expanduser().resolve() if supplied_seg else output / "seg_manifest.json"
    if supplied_seg and not seg_manifest.is_file():
        raise FileNotFoundError(f"Native predicted head-segmentation manifest required: {seg_manifest}")
    config = OmegaConf.load(template)
    train = config.data.train
    train.episode_paths = episodes
    train.prompt_index_file = str(output / "prompt_index.json")
    train.text_embedding_cache_dir = str(Path(text_cache).expanduser().resolve())
    train.seg_quality_mix_manifest = str(seg_manifest)
    if not train.strict_pred_head_only:
        raise ValueError("The main training recipe requires predicted head segmentation")
    output.mkdir(parents=True)
    write_json(output / "prompt_index.json", dict(schema_version=PROMPT_INDEX_SCHEMA_VERSION,
               record_count=len(records), records=list(records.values())))
    prompts = sorted({DEFAULT_PROMPT.format(task=r["instruction"]) for r in records.values()})
    write_json(output / "prompts.json", prompts)
    write_json(output / "perception_manifest.json", dict(schema_version="fastwam.robotwin-predicted-perception-pilot.v1",
        status="PASS", scope="Per-episode RGB and instruction manifest",
        aggregate=dict(selected_episode_count=len(perception_rows)), episodes=perception_rows))
    OmegaConf.save(config, output / "data.yaml")
    result = dict(status="DATA_MANIFEST_PREPARED", episode_count=len(episodes), prompt_count=len(prompts),
                  data_config=str(output / "data.yaml"), source_manifest=str(manifest),
                  note="Window indexing, text-cache and per-frame segmentation coverage are checked by check-data.")
    write_json(output / "summary.json", result)
    return result


def index_segmentation(manifest_path, cache_dir, output):
    """Index the predicted segmentation masks for each episode."""
    import hashlib
    import h5py
    import numpy as np
    from .datasets.robotwin_momap.source_input_augmentation import (
        SEG_CACHE_SCHEMA, SEG_QUALITY_MIX_MANIFEST_SCHEMA, _canonical_digest,
    )
    rows = json.loads(Path(manifest_path).read_text())["episodes"]
    entries, math_rows = {}, []
    for row in rows:
        episode_id = row["episode_id"]
        key = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
        path = Path(cache_dir).resolve() / "episodes" / f"{key}.h5"
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != SEG_CACHE_SCHEMA or handle.attrs.get("status") != "PASS":
                raise ValueError(f"Invalid/incomplete predicted segmentation cache: {path}")
            if handle.attrs.get("episode_id") != episode_id:
                raise ValueError(f"Segmentation episode mismatch: {path}")
            source = row["source"]
            expected = np.arange(source["legal_source_index_start"], source["legal_source_index_stop_inclusive"] + 1)
            if not np.array_equal(handle["source_index"][:], expected):
                raise ValueError(f"Missing segmentation frame coverage: {path}")
        entries[episode_id] = dict(cache_path=str(path), p_gt=0.0, quality_bucket="predicted_only_unscored")
        math_rows.append((episode_id, 0.0, "predicted_only_unscored"))
    result = dict(schema_version=SEG_QUALITY_MIX_MANIFEST_SCHEMA, episode_count=len(entries),
                  policy_content_digest=_canonical_digest(sorted(math_rows)), episodes=entries,
                  note="Per-frame predicted head-camera segmentation masks.")
    write_json(output, result)
    return dict(status="SEGMENTATION_INDEX_COMPLETE", episode_count=len(entries))


def encode_text(prompts_path, cache_dir, weights=None, tokenizer_path=None, device="cuda", batch_size=1):
    import torch
    from .datasets.robotwin_momap.text_cache import CachedTextContextEncoder
    prompts = json.loads(Path(prompts_path).read_text())
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("prompts.json must contain a nonempty list of formatted prompt strings")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = CachedTextContextEncoder(cache_dir)
    todo = []
    for prompt in sorted(set(prompts)):
        if cache.cache_path(prompt).exists():
            cache(prompt)
        else:
            todo.append(prompt)
    if todo:
        if weights is None or tokenizer_path is None or not Path(weights).is_file() or not Path(tokenizer_path).is_dir():
            raise FileNotFoundError(f"{len(todo)} prompts need local Wan T5 weights and tokenizer")
        from .models.wan22.helpers.loader import load_text_encoder
        from .models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
        model = load_text_encoder(str(weights), device)
        tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=128, clean="whitespace")
        with torch.inference_mode():
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
                context = model(ids.to(device), mask.to(device=device, dtype=torch.bool))
                for i, prompt in enumerate(batch):
                    payload = dict(context=context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                                   mask=mask[i].to(device="cpu", dtype=torch.bool).contiguous())
                    target = cache.cache_path(prompt)
                    temporary = target.with_suffix(f".tmp.{os.getpid()}")
                    torch.save(payload, temporary)
                    os.replace(temporary, target)
                    cache(prompt)
                print(f"Encoded {min(start + batch_size, len(todo))}/{len(todo)} prompts", flush=True)
    return dict(status="TEXT_CACHE_COMPLETE", prompts=len(set(prompts)), newly_encoded=len(todo), cache_dir=str(cache_dir))


def check_data(config_path, output, batches=1, batch_size=1, workers=0):
    import torch
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from .utils.pytorch_utils import set_global_seed
    if batches < 1 or batch_size < 1 or workers < 0:
        raise ValueError("batches/batch_size must be positive; workers must be nonnegative")
    set_global_seed(43)
    dataset = instantiate(OmegaConf.load(config_path).data.train)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=dataset.collate_fn)
    reports = []
    for index, batch in enumerate(loader):
        tensors = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite training tensor: {key}")
                tensors[key] = dict(shape=list(value.shape), dtype=str(value.dtype))
        reports.append(tensors)
        if index + 1 >= batches:
            break
    if not reports:
        raise ValueError("No valid training windows")
    result = dict(status="REAL_DATALOADER_PASSED", dataset_size=len(dataset), batches_checked=len(reports),
                  tensors=reports, scope="Only the requested batches were read; not an exhaustive corpus check.")
    write_json(output, result)
    return result
