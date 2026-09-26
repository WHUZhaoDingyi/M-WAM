"""Method-agnostic orchestration for RoboTwin MoMap training samples.

The dataset deliberately does not know how an HDF5 episode is encoded or how
3D flow is computed.  Those responsibilities are injected through
``CanonicalSampleSource`` and ``MoMapTargetBuilder``.  Source-time GT
depth/seg-derived conditions are explicit, audited ``ModelInputs`` fields.
Keeping future privileged geometry inside the target-builder boundary
prevents look-ahead data from becoming a model input by accident.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from mwam.evaluation.contracts import (
    STANDALONE_MOMAP_CHECKPOINT_EVALUATOR,
)

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


def _stable_callable_identifier(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        adapter_type = type(value)
        module = adapter_type.__module__
        qualname = adapter_type.__qualname__
    if (
        not module
        or not qualname
        or module == "__main__"
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        raise ValueError(
            "target_transform must have a stable qualified name"
        )
    return f"{module}.{qualname}"


def _reject_provenance_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(
                fragment in normalized
                for fragment in ("path", "sha", "receipt")
            ):
                raise ValueError(
                    "resume math contract must not contain path/SHA/receipt keys"
                )
            _reject_provenance_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_provenance_keys(nested)


class RoboTwinMoMapDataset(Dataset):
    """Compose model-visible inputs and labels for one indexed sample.

    ``source`` owns the concrete storage implementation and worker-local file
    handles.  ``target_builder`` is the only component allowed to consume
    ``PrivateGeometry``.  Neither object is interpreted by this class.
    """

    in_trainer_evaluation_contract = STANDALONE_MOMAP_CHECKPOINT_EVALUATOR

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

    def resume_math_contract(self) -> dict[str, Any]:
        """Return finite JSON mathematics needed for exact retraining."""

        source_provider = getattr(
            self.source,
            "resume_math_contract",
            None,
        )
        if not callable(source_provider):
            raise TypeError(
                "source must provide resume_math_contract()"
            )
        source_contract = source_provider()
        if not isinstance(source_contract, Mapping):
            raise TypeError(
                "source resume_math_contract() must return a mapping"
            )

        binding_provider = getattr(
            self.target_builder,
            "resume_binding",
            None,
        )
        if not callable(binding_provider):
            raise TypeError(
                "target_builder must provide resume_binding()"
            )
        binding = binding_provider()
        if not isinstance(binding, Mapping):
            raise TypeError(
                "target_builder resume_binding() must return a mapping"
            )
        target_fields = (
            "target_builder_adapter",
            "target_service_factory",
            "grid_size",
            "output_dtype",
            "evaluate_visibility",
            "depth_tolerance_m",
            "pixel_boundary_ambiguity_px",
            "reference_camera_name",
            "se3_validation_schema_version",
        )
        missing = [field for field in target_fields if field not in binding]
        if missing:
            raise ValueError(
                "target_builder resume binding is missing math fields: "
                f"{missing}"
            )
        target_contract = {
            field: binding[field] for field in target_fields
        }
        # Historical injected builders predate explicit representation.  Their
        # only legal interpretation is the legacy all-view canonical XYZ
        # target; production hybrid builders publish the field explicitly.
        target_contract["target_representation"] = binding.get(
            "target_representation",
            "canonical_dxyz_all_views_v1",
        )
        if "wrist_flow_max_magnitude_native_px" in binding:
            target_contract["wrist_flow_max_magnitude_native_px"] = binding[
                "wrist_flow_max_magnitude_native_px"
            ]
            target_contract["momap_loss_valid_semantics"] = binding[
                "momap_loss_valid_semantics"
            ]
        if binding.get("wrist_flow_gt_background_filter") is True:
            for field in ("wrist_flow_gt_background_filter", "momap_loss_valid_semantics",
                          "gt_background_entity_names"):
                target_contract[field] = binding[field]
        transform_identifier = (
            None
            if self.target_transform is None
            else _stable_callable_identifier(self.target_transform)
        )
        contract: dict[str, Any] = {
            "schema_version": "robotwin_momap_dataset_math_v1",
            "source": dict(source_contract),
            "target_builder": target_contract,
            "target_transform": transform_identifier,
        }
        _reject_provenance_keys(contract)
        try:
            json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "resume math contract must be finite pure JSON"
            ) from exc
        return contract


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
