"""Method-agnostic orchestration for RoboTwin MoMap training samples.

The dataset deliberately does not know how an HDF5 episode is encoded or how
3D flow is computed.  Those responsibilities are injected through
``CanonicalSampleSource`` and ``MoMapTargetBuilder``.  Source-time GT
depth/seg-derived conditions are explicit, audited ``ModelInputs`` fields.
Keeping future privileged geometry inside the target-builder boundary
prevents look-ahead data from becoming a model input by accident.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset


from .collator import MoMapCollator
from .interfaces import CanonicalSampleSource, MoMapTargetBuilder
from .schema import (
    MoMapTarget,
    SampleRef,
    training_sample_to_dict,
    validate_training_sample,
)

TargetTransform = Callable[[MoMapTarget], MoMapTarget]


def finiteize_invalid_momap(target: MoMapTarget) -> MoMapTarget:
    """Replace invalid MoMap values with zero without changing mask semantics.

    Upstream geometry code may use NaN to represent an invalid point.  NaN
    cannot enter a diffusion scheduler because ``NaN * 0`` is still NaN.  A
    non-finite value at a *valid* point remains a hard error.
    """

    valid_xyz = (
        target.geom_valid.expand_as(target.momap)
        if target.component_valid is None
        else target.component_valid
    )
    finite = torch.isfinite(target.momap)
    if bool((valid_xyz & ~finite).any().item()):
        raise ValueError("momap contains NaN/Inf at geom_valid locations")

    finite_momap = torch.where(
        valid_xyz,
        target.momap,
        torch.zeros((), dtype=target.momap.dtype, device=target.momap.device),
    )
    return replace(target, momap=finite_momap)


class RoboTwinMoMapDataset(Dataset):
    """Compose model-visible inputs and labels for one indexed sample.

    ``source`` owns the concrete storage implementation and worker-local file
    handles.  ``target_builder`` is the only component allowed to consume
    ``PrivateGeometry``.  Neither object is interpreted by this class.
    """


    def __init__(
        self,
        source: CanonicalSampleSource,
        target_builder: MoMapTargetBuilder,
        *,
        target_transform: TargetTransform | None = None,
        collator: MoMapCollator | None = None,
    ) -> None:
        self.source = source
        self.target_builder = target_builder
        self.target_transform = target_transform
        self._collator = collator or MoMapCollator()

        sample_refs: Sequence[SampleRef] = source.sample_refs
        self._sample_refs = sample_refs
        if len(self._sample_refs) <= 0:
            raise ValueError("RoboTwinMoMapDataset requires at least one SampleRef")
        probes = {0, len(self._sample_refs) - 1}
        if not all(isinstance(self._sample_refs[index], SampleRef) for index in probes):
            raise TypeError("source.sample_refs must resolve SampleRef values")

    @property
    def sample_refs(self) -> Sequence[SampleRef]:
        return self._sample_refs

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch state used by deterministic source augmentation."""

        setter = getattr(self.source, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    @property
    def requires_fork_cow_workers(self) -> bool:
        """Forward the optional text-bank worker-start contract."""

        return bool(
            getattr(self.source, "requires_fork_cow_workers", False)
        )


    def episode_index_ranges(self) -> tuple[tuple[int, int], ...]:
        """Expose immutable per-episode ranges without materializing samples."""

        provider = getattr(self._sample_refs, "episode_index_ranges", None)
        if not callable(provider):
            raise TypeError(
                "dataset sample_refs do not expose episode_index_ranges()"
            )
        ranges = provider()
        if not isinstance(ranges, tuple):
            raise TypeError("episode_index_ranges() must return a tuple")
        return ranges

    def episode_index_ranges_binding(self) -> dict[str, object]:
        """Expose the compact episode-range identity used by the sampler."""

        provider = getattr(
            self._sample_refs,
            "episode_index_ranges_binding",
            None,
        )
        if not callable(provider):
            raise TypeError(
                "dataset sample_refs do not expose "
                "episode_index_ranges_binding()"
            )
        binding = provider()
        if not isinstance(binding, Mapping):
            raise TypeError(
                "episode_index_ranges_binding() must return a mapping"
            )
        return dict(binding)


    @property
    def collate_fn(self) -> MoMapCollator:
        """Return the strict collator to pass to ``torch.utils.data.DataLoader``."""

        return self._collator

    def __len__(self) -> int:
        return len(self._sample_refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_ref = self._sample_refs[index]
        model_inputs = self.source.load_model_inputs(sample_ref)
        action_target = self.source.load_action_target(sample_ref)

        # Future/private target-construction state exists only across this
        # builder call.  It is not merged into ``sample`` and is explicitly
        # released before validation.
        private_geometry = self.source.load_private_geometry(sample_ref)
        try:
            momap_target = self.target_builder.build_target(
                sample_ref,
                private_geometry,
            )
        finally:
            del private_geometry

        momap_target = finiteize_invalid_momap(momap_target)
        if self.target_transform is not None:
            momap_target = self.target_transform(momap_target)
            if not isinstance(momap_target, MoMapTarget):
                raise TypeError("target_transform must return MoMapTarget")

        # Re-check after a transform so normalization cannot introduce NaN/Inf
        # at valid locations and invalid locations remain scheduler-safe.
        momap_target = finiteize_invalid_momap(momap_target)
        sample = training_sample_to_dict(
            sample_ref,
            model_inputs,
            action_target,
            momap_target,
        )
        validate_training_sample(sample)
        return sample
