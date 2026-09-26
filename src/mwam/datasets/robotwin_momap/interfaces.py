"""Replaceable storage and target-construction interfaces for MoMap data."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .schema import ActionTarget, ModelInputs, MoMapTarget, PrivateGeometry, SampleRef


@runtime_checkable
class CanonicalSampleSource(Protocol):
    """Read the three ownership classes for an already-resolved sample.

    The concrete V1 implementation owns worker-local HDF5 handles.  A source
    Source-time oracle conditions are explicit fields of ``ModelInputs``.
    Future geometry and all target-construction-only simulator state may
    leave the source only through ``PrivateGeometry``.
    """

    @property
    def sample_refs(self) -> Sequence[SampleRef]: ...

    def load_model_inputs(self, sample_ref: SampleRef) -> ModelInputs: ...

    def load_action_target(self, sample_ref: SampleRef) -> ActionTarget: ...

    def load_private_geometry(self, sample_ref: SampleRef) -> PrivateGeometry: ...


@runtime_checkable
class MoMapTargetBuilder(Protocol):
    """Convert teacher-only geometry into model-facing MoMap supervision."""

    def build_target(
        self,
        sample_ref: SampleRef,
        geometry: PrivateGeometry,
    ) -> MoMapTarget: ...
