# Example observations

Use the documented `example-input` command to create a deterministic NPZ with all seven input tensors. It is **synthetic smoke-test data**, not a demonstration or a robotics evaluation scene.

For real inference, export a current observation following `docs/implementation.md`; keep future labels out of the file. Text must use the checkpoint-compatible 4096-wide language embeddings. Raw depth and segmentation IDs are not interchangeable with the prepared XYZ and segmentation condition tensors.

Output `action_raw` has shape `[B,32,14]`. The interface does not send it to a robot.
