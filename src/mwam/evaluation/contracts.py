"""Small capability markers shared by datasets and the generic trainer.

The markers describe which evaluator owns a dataset.  They deliberately carry
no model, Dataset, simulator, or Formal imports so checking the boundary stays
safe during lightweight startup.
"""

LEGACY_VIDEO_PROMPT_EVALUATOR = "legacy_video_prompt_v1"
STANDALONE_MOMAP_CHECKPOINT_EVALUATOR = "standalone_momap_checkpoint_v1"


__all__ = [
    "LEGACY_VIDEO_PROMPT_EVALUATOR",
    "STANDALONE_MOMAP_CHECKPOINT_EVALUATOR",
]
