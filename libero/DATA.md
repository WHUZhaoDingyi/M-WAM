# Training input

The LIBERO reference loader consumes exported simulator episodes. Prepare a JSON index of the episodes on your filesystem. Paths in this index are relative to the index file or absolute. List the task's objects of interest using the names from its BDDL file. Supply the same official instruction used to make the track and language caches.

```json
{
  "text_cache": "gemma",
  "episodes": [
    {
      "geometry": "libero_goal/put_the_bowl_on_the_stove_demo/demo_0/episode.hdf5",
      "proprio": "proprio/demo_0.npy",
      "segmentation": "masks/demo_0.h5",
      "instruction": "put the bowl on the stove",
      "task_objects": ["akita_black_bowl_1", "flat_stove_1"]
    }
  ]
}
```

The example shows one record. Supply at least two independent demonstrations per task. The reference run used 50 demonstrations for each of 40 tasks. The preparer uses the recorded suite, task and demo identity and the seed-43 per-task split (45 training and five held out in the reference run).

| Input | Expected content |
| --- | --- |
| Geometry HDF5 | A 20 Hz M-WAM geometry export with the `agentview` and `robot0_eye_in_hand` cameras, RGB rasters, metric geometry, source pixel entity IDs, camera-to-world transforms, entity registry and per-frame entity poses/validity, timeline, and the demonstrated `robot_state/action_target[T,7]`. The reader supports the encodings in [raw_episode_hdf5.py](src/mwam/robotwin_momap/raw_episode_hdf5.py). `metadata/episode_json` identifies the BDDL file and the source demo. |
| Proprioception | A NumPy file `[T,8]` with the official observation at each source frame: end-effector XYZ, axis-angle rotation from the signed body quaternion, and the two finger joint positions. |
| Predicted segmentation | A causal DINO+SAM2 head-camera cache in the native RGB raster. The HDF5 contains `source_index[N]`, `camera_name=["agentview"]`, and integer `pred/track_id[N,1,H,W]`. Track IDs are recolored by the dataset on each clip draw. Cache metadata binds it to the source geometry, instruction, camera and pixel convention; see [segmentation.py](src/mwam/datasets/libero_momap/segmentation.py). |
| Text | The versioned Gemma-2-2b-it context cache: `encoder_contract.json` and one `<sha256 of UTF-8 instruction>.gemma2_len300.pt` per instruction. Each file contains `context[300,2304]` and boolean `mask[300]`. See [gemma_cache.py](src/mwam/datasets/robotwin_momap/gemma_cache.py). |

From an installed checkout, run:

```bash
python prepare_data.py --index /path/to/input.json --out /path/to/ready
```

This writes the episode manifest, segmentation manifest, and normalization used by the trainer. Action and proprio min/max cover all real training actions and indexed training states. Geometry uses 512 training windows sampled with seed 43. On-demand target generation computes cumulative XYZ displacements at +8/+16/+24/+32 steps.

The index is portable. The generated manifests bind the paths on the current machine; after moving inputs, run the preparer again with a new output directory. For moving historical track caches, preserve geometry file modification times and the `suite/task_demo/demo_n/episode.hdf5` suffix. New cache exporters can record the geometry file's SHA-256 in the `source_geometry_sha256` HDF5 attribute. For each demonstration, provide the geometry export and track cache alongside its instruction embedding and robot state file.
