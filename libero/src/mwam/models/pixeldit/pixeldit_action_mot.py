from __future__ import annotations


import torch.nn as nn


class PixelDiTActionMoT(nn.Module):
    """Versioned one-way MoMap-to-Action lockstep architectures."""

    INTERACTION_MODES = frozenset({"none", "momap_to_action"})


    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    @interaction_mode.setter
    def interaction_mode(self, value: str) -> None:
        if value not in self.INTERACTION_MODES:
            raise ValueError(
                f"`interaction_mode` must be one of "
                f"{sorted(self.INTERACTION_MODES)}, got {value!r}."
            )
        self._interaction_mode = value

    @property
    def bridge_semantics_version(self) -> str:
        """Immutable execution-order identity bound into training checkpoints."""

        return self._bridge_semantics_version

    @property
    def topology_version(self) -> str:
        """Immutable topology identity bound into training checkpoints."""

        return self._topology_version


    @property
    def layer_map(self) -> tuple[int, ...]:
        """Compatibility metadata; Action execution is identity by topology."""

        return tuple(range(self.action_depth))

