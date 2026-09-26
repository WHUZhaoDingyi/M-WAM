from __future__ import annotations

import hashlib
import json
from math import ceil
from typing import Any, Iterator, Mapping, Sized

import torch
from torch.utils.data import Sampler


RESUMABLE_EPOCH_SAMPLER_SCHEMA_VERSION = "fastwam_resumable_epoch_sampler_v3"
LEGACY_RESUMABLE_EPOCH_SAMPLER_SCHEMA_VERSION = (
    "fastwam_legacy_resumable_epoch_sampler_v1"
)
LEGACY_UNPADDED_TAIL_POLICY = (
    "legacy_unpadded_dataset_permutation_allow_partial_final_update"
)
RESUMABLE_EPISODE_WINDOW_SAMPLER_SCHEMA_VERSION = (
    "fastwam_resumable_episode_window_sampler_v1"
)
INITIAL_WINDOW_PREFERRED_SAMPLER_SCHEMA_VERSION = (
    "fastwam_initial_window_preferred_sampler_v1"
)
FAILURE_REPLAY_SAMPLER_SCHEMA_VERSION = (
    "fastwam_failure_replay_sampler_v1"
)
FAILURE_REPLAY_MANIFEST_SCHEMA_VERSION = (
    "fastwam_failure_replay_manifest_v1"
)
FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_SAMPLER_SCHEMA_VERSION = (
    "fastwam_failure_replay_task_variant_noinitial_sampler_v1"
)
FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_MANIFEST_SCHEMA_VERSION = (
    "fastwam_failure_replay_task_variant_noinitial_manifest_v1"
)
ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH = (
    "one_random_window_per_episode_per_epoch"
)
INITIAL_WINDOW_10PCT_REMAINING_90PCT = (
    "initial_window_10pct_remaining_90pct"
)
FAILURE_REPLAY_80PCT_INITIAL10 = "failure_replay_80pct_initial10"
FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL = (
    "failure_replay_80pct_task_variant_noinitial"
)


def _canonical_sha256(value: object, *, prefix: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(b"\0")
    digest.update(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


class ResumableEpochSampler(Sampler[int]):
    """Preserve the original FastWAM unpadded epoch stream.

    Generic FastWAM training shuffles each dataset item exactly once per epoch.
    MoMap production can opt into :class:`StrictResumableEpochSampler` when it
    needs effective-global-batch padding and exact optimizer-boundary resume.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if len(self.dataset) <= 0:
            raise ValueError("ResumableEpochSampler requires a non-empty dataset")
        if (
            self.batch_size <= 0
            or self.num_processes <= 0
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "batch_size, num_processes, and gradient_accumulation_steps "
                "must be positive"
            )
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._resume_effective_epoch: int | None = None

    @property
    def global_microbatch_size(self) -> int:
        return self.batch_size * self.num_processes

    @property
    def global_batch_size(self) -> int:
        return self.global_microbatch_size

    @property
    def effective_global_batch_size(self) -> int:
        return self.global_microbatch_size * self.gradient_accumulation_steps

    @property
    def total_size(self) -> int:
        return len(self.dataset)

    @property
    def global_batches_per_epoch(self) -> int:
        return ceil(len(self.dataset) / self.global_microbatch_size)

    @property
    def optimizer_updates_per_epoch(self) -> int:
        return ceil(
            self.global_batches_per_epoch / self.gradient_accumulation_steps
        )

    @property
    def effective_epoch(self) -> int:
        return self.epoch + self.epoch_offset

    def binding(self) -> dict[str, int | str]:
        return {
            "schema_version": LEGACY_RESUMABLE_EPOCH_SAMPLER_SCHEMA_VERSION,
            "dataset_length": len(self.dataset),
            "seed": self.seed,
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_microbatch_size": self.global_microbatch_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "stream_samples_per_epoch": self.total_size,
            "global_batches_per_epoch": self.global_batches_per_epoch,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "tail_policy": LEGACY_UNPADDED_TAIL_POLICY,
        }

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)
        if self.epoch_offset < 0:
            raise ValueError("epoch_offset must be non-negative")

    def set_resume_batch_offset(self, batch_in_epoch: int):
        offset = int(batch_in_epoch)
        if offset < 0 or offset > self.global_batches_per_epoch:
            raise ValueError(
                "resume batch offset is outside the unpadded epoch: "
                f"{offset} not in [0,{self.global_batches_per_epoch}]"
            )
        self.resume_batch_offset = offset
        self._resume_effective_epoch = self.effective_epoch

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0
        self._resume_effective_epoch = None

    def _sample_offset(self) -> int:
        if (
            self.resume_batch_offset > 0
            and self._resume_effective_epoch == self.effective_epoch
        ):
            return self.resume_batch_offset * self.global_microbatch_size
        return 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.effective_epoch)
        indices = torch.randperm(
            len(self.dataset), generator=generator
        ).tolist()
        return iter(indices[self._sample_offset() :])

    def __len__(self) -> int:
        return max(len(self.dataset) - self._sample_offset(), 0)


class StrictResumableEpochSampler(Sampler[int]):
    """Deterministic global stream padded to complete optimizer updates.

    Accelerate shards consecutive local batches across ranks.  Padding the
    permutation to a multiple of the *effective* global batch before applying
    the resume cursor guarantees that the final optimizer update of every
    epoch contains exactly gradient_accumulation_steps microbatches.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if len(self.dataset) <= 0:
            raise ValueError(
                "StrictResumableEpochSampler requires a non-empty dataset"
            )
        if (
            self.batch_size <= 0
            or self.num_processes <= 0
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "batch_size, num_processes, and gradient_accumulation_steps "
                "must be positive"
            )
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._resume_effective_epoch: int | None = None

    @property
    def global_microbatch_size(self) -> int:
        return self.batch_size * self.num_processes

    @property
    def global_batch_size(self) -> int:
        """Backward-compatible alias for a simultaneous global microbatch."""

        return self.global_microbatch_size

    @property
    def effective_global_batch_size(self) -> int:
        return self.global_microbatch_size * self.gradient_accumulation_steps

    @property
    def total_size(self) -> int:
        return (
            ceil(len(self.dataset) / self.effective_global_batch_size)
            * self.effective_global_batch_size
        )

    @property
    def global_batches_per_epoch(self) -> int:
        return self.total_size // self.global_microbatch_size

    @property
    def optimizer_updates_per_epoch(self) -> int:
        return self.global_batches_per_epoch // self.gradient_accumulation_steps

    @property
    def effective_epoch(self) -> int:
        return self.epoch + self.epoch_offset

    def binding(self) -> dict[str, int | str]:
        return {
            "schema_version": RESUMABLE_EPOCH_SAMPLER_SCHEMA_VERSION,
            "dataset_length": len(self.dataset),
            "seed": self.seed,
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_microbatch_size": self.global_microbatch_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "padded_samples_per_epoch": self.total_size,
            "global_batches_per_epoch": self.global_batches_per_epoch,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "tail_policy": (
                "repeat_epoch_prefix_to_effective_global_batch_multiple"
            ),
        }

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)
        if self.epoch_offset < 0:
            raise ValueError("epoch_offset must be non-negative")

    def set_resume_batch_offset(self, batch_in_epoch: int):
        offset = int(batch_in_epoch)
        if offset < 0 or offset > self.global_batches_per_epoch:
            raise ValueError(
                "resume batch offset is outside the padded epoch: "
                f"{offset} not in [0,{self.global_batches_per_epoch}]"
            )
        if offset % self.gradient_accumulation_steps != 0:
            raise ValueError(
                "resume batch offset must be an optimizer-update boundary: "
                f"{offset} is not divisible by {self.gradient_accumulation_steps}"
            )
        self.resume_batch_offset = offset
        self._resume_effective_epoch = self.effective_epoch

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0
        self._resume_effective_epoch = None

    def _epoch_indices(self) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.effective_epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        padding = self.total_size - len(indices)
        if padding > 0:
            repeats = ceil(padding / len(indices))
            indices.extend((indices * repeats)[:padding])
        return indices

    def _sample_offset(self) -> int:
        if (
            self.resume_batch_offset > 0
            and self._resume_effective_epoch == self.effective_epoch
        ):
            return self.resume_batch_offset * self.global_microbatch_size
        return 0

    def __iter__(self) -> Iterator[int]:
        return iter(self._epoch_indices()[self._sample_offset() :])

    def __len__(self) -> int:
        return self.total_size - self._sample_offset()


class ResumableEpisodeWindowSampler(StrictResumableEpochSampler):
    """Select one uniformly random legal window from every episode per epoch.

    The dataset index remains the immutable address table for all legal
    windows. At each effective epoch this sampler chooses one address from
    every episode's half-open global index range, globally shuffles those
    choices, and pads by repeating the shuffled prefix so every optimizer
    update is complete. All ranks construct the same global stream; the
    downstream Accelerate batch sharder assigns rank-local batches.

    Stateful randomness intentionally lives here rather than in
    Dataset.__getitem__. This makes the selected stream a pure function of the
    frozen episode ranges, seed, and effective_epoch and allows exact
    optimizer-boundary resume.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self._episode_ranges = self._load_episode_ranges(dataset)
        self._episode_ranges_binding = self._load_episode_ranges_binding(
            dataset
        )
        self._episode_range_boundaries_sha256 = _canonical_sha256(
            {
                "dataset_length": len(self.dataset),
                "ranges": self._episode_ranges,
            },
            prefix=b"fastwam-episode-index-range-boundaries-v1",
        )

    @staticmethod
    def _load_episode_ranges(dataset: Sized) -> tuple[tuple[int, int], ...]:
        provider = getattr(dataset, "episode_index_ranges", None)
        if not callable(provider):
            raise TypeError(
                "episode-aware sampling requires dataset.episode_index_ranges()"
            )
        raw_ranges = provider()
        if not isinstance(raw_ranges, (tuple, list)) or not raw_ranges:
            raise ValueError("episode_index_ranges() must return non-empty ranges")

        ranges: list[tuple[int, int]] = []
        expected_start = 0
        for position, value in enumerate(raw_ranges):
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(
                    "each episode index range must be a (start, stop) pair"
                )
            start, stop = value
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(stop, bool)
                or not isinstance(stop, int)
            ):
                raise TypeError("episode index range bounds must be integers")
            if start != expected_start:
                raise ValueError(
                    "episode index ranges must be contiguous and ordered: "
                    f"range {position} starts at {start}, expected {expected_start}"
                )
            if stop <= start:
                raise ValueError("every episode index range must be non-empty")
            ranges.append((start, stop))
            expected_start = stop

        if expected_start != len(dataset):
            raise ValueError(
                "episode index ranges must exactly cover the dataset: "
                f"covered [0,{expected_start}), dataset length is {len(dataset)}"
            )
        return tuple(ranges)

    def _load_episode_ranges_binding(self, dataset: Sized) -> dict[str, Any]:
        provider = getattr(dataset, "episode_index_ranges_binding", None)
        if not callable(provider):
            raise TypeError(
                "episode-aware sampling requires "
                "dataset.episode_index_ranges_binding()"
            )
        raw = provider()
        if not isinstance(raw, Mapping):
            raise TypeError("episode_index_ranges_binding() must return a mapping")
        required = {
            "schema_version",
            "digest_algorithm",
            "episode_count",
            "sample_count",
            "ranges_sha256",
        }
        if set(raw) != required:
            raise ValueError(
                "episode index ranges binding keys drifted: "
                f"missing={sorted(required - set(raw))}, "
                f"unexpected={sorted(set(raw) - required)}"
            )
        if not isinstance(raw["schema_version"], str) or not raw[
            "schema_version"
        ]:
            raise ValueError("episode index ranges binding schema is invalid")
        if not isinstance(raw["digest_algorithm"], str) or not raw[
            "digest_algorithm"
        ]:
            raise ValueError("episode index ranges digest algorithm is invalid")
        if raw["episode_count"] != len(self._episode_ranges):
            raise ValueError("episode index ranges binding episode_count drifted")
        if raw["sample_count"] != len(dataset):
            raise ValueError("episode index ranges binding sample_count drifted")
        if not _is_sha256(raw["ranges_sha256"]):
            raise ValueError(
                "episode index ranges binding ranges_sha256 is invalid"
            )
        return dict(raw)

    @property
    def selected_samples_per_epoch(self) -> int:
        return len(self._episode_ranges)

    @property
    def total_size(self) -> int:
        return (
            ceil(
                self.selected_samples_per_epoch
                / self.effective_global_batch_size
            )
            * self.effective_global_batch_size
        )

    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": (
                RESUMABLE_EPISODE_WINDOW_SAMPLER_SCHEMA_VERSION
            ),
            "sampling_mode": ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH,
            "dataset_length": len(self.dataset),
            "episode_count": len(self._episode_ranges),
            "clips_per_episode": 1,
            # Concrete half-open ranges determine the sampled index stream.
            # The legacy digests below remain only for v4 compatibility and
            # are filtered out of newly written resume metadata.
            "episode_ranges": [
                [int(start), int(stop)]
                for start, stop in self._episode_ranges
            ],
            "episode_ranges_binding_schema_version": str(
                self._episode_ranges_binding["schema_version"]
            ),
            "episode_ranges_digest_algorithm": str(
                self._episode_ranges_binding["digest_algorithm"]
            ),
            "episode_ranges_identity_sha256": str(
                self._episode_ranges_binding["ranges_sha256"]
            ),
            "episode_range_boundaries_sha256": (
                self._episode_range_boundaries_sha256
            ),
            "seed": self.seed,
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_microbatch_size": self.global_microbatch_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "selected_samples_per_epoch": self.selected_samples_per_epoch,
            "padded_samples_per_epoch": self.total_size,
            "padding_samples_per_epoch": (
                self.total_size - self.selected_samples_per_epoch
            ),
            "global_batches_per_epoch": self.global_batches_per_epoch,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "selection_policy": "uniform_integer_within_each_half_open_range",
            "shuffle_policy": "global_randperm_after_episode_selection",
            "tail_policy": (
                "repeat_shuffled_epoch_prefix_to_effective_global_batch_multiple"
            ),
        }

    def _epoch_indices(self) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.effective_epoch)
        selected = [
            start
            + int(
                torch.randint(
                    stop - start,
                    (1,),
                    generator=generator,
                ).item()
            )
            for start, stop in self._episode_ranges
        ]
        permutation = torch.randperm(
            len(selected), generator=generator
        ).tolist()
        indices = [selected[position] for position in permutation]
        padding = self.total_size - len(indices)
        if padding > 0:
            repeats = ceil(padding / len(indices))
            indices.extend((indices * repeats)[:padding])
        return indices


class InitialWindowPreferredSampler(ResumableEpisodeWindowSampler):
    """Build a strict epoch stream with 10% episode-initial windows.

    Each episode's half-open index range begins at its first legal training
    window.  This sampler reserves the nearest integer to 10% of the strict
    padded epoch stream for those range starts and fills the remaining 90%
    from all other legal indices.  Sampling inside either pool cycles through
    independent permutations, so no item repeats until its pool is exhausted.
    The two pools are then globally shuffled.

    The resulting stream length is exactly the same as
    :class:`StrictResumableEpochSampler` for the same dataset and batch
    topology.  It therefore preserves optimizer-update counts and inherits
    the same optimizer-boundary resume cursor semantics.
    """

    _INITIAL_NUMERATOR = 1
    _INITIAL_DENOMINATOR = 10

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        initial_pool = torch.tensor(
            [start for start, _ in self._episode_ranges],
            dtype=torch.long,
        )
        remaining_mask = torch.ones(len(self.dataset), dtype=torch.bool)
        remaining_mask[initial_pool] = False
        remaining_pool = torch.arange(len(self.dataset), dtype=torch.long)[
            remaining_mask
        ]
        if initial_pool.numel() == 0 or remaining_pool.numel() == 0:
            raise ValueError(
                "initial-window-preferred sampling requires both initial "
                "and non-initial legal windows"
            )
        self._initial_pool = initial_pool
        self._remaining_pool = remaining_pool

    @property
    def total_size(self) -> int:
        return (
            ceil(len(self.dataset) / self.effective_global_batch_size)
            * self.effective_global_batch_size
        )

    @property
    def selected_samples_per_epoch(self) -> int:
        return self.total_size

    @property
    def initial_samples_per_epoch(self) -> int:
        numerator = self.total_size * self._INITIAL_NUMERATOR
        return (numerator + self._INITIAL_DENOMINATOR // 2) // (
            self._INITIAL_DENOMINATOR
        )

    @property
    def remaining_samples_per_epoch(self) -> int:
        return self.total_size - self.initial_samples_per_epoch

    @staticmethod
    def _balanced_pool_sample(
        pool: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample ``count`` values, exhausting shuffled pools before repeats."""

        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        pool_size = int(pool.numel())
        full_cycles, tail = divmod(int(count), pool_size)
        chunks = [
            pool[torch.randperm(pool_size, generator=generator)]
            for _ in range(full_cycles)
        ]
        if tail:
            chunks.append(
                pool[torch.randperm(pool_size, generator=generator)[:tail]]
            )
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": INITIAL_WINDOW_PREFERRED_SAMPLER_SCHEMA_VERSION,
            "sampling_mode": INITIAL_WINDOW_10PCT_REMAINING_90PCT,
            "dataset_length": len(self.dataset),
            "episode_count": len(self._episode_ranges),
            "episode_ranges": [
                [int(start), int(stop)]
                for start, stop in self._episode_ranges
            ],
            "episode_ranges_binding_schema_version": str(
                self._episode_ranges_binding["schema_version"]
            ),
            "episode_ranges_digest_algorithm": str(
                self._episode_ranges_binding["digest_algorithm"]
            ),
            "episode_ranges_identity_sha256": str(
                self._episode_ranges_binding["ranges_sha256"]
            ),
            "episode_range_boundaries_sha256": (
                self._episode_range_boundaries_sha256
            ),
            "seed": self.seed,
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_microbatch_size": self.global_microbatch_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "initial_fraction_numerator": self._INITIAL_NUMERATOR,
            "initial_fraction_denominator": self._INITIAL_DENOMINATOR,
            "initial_pool_size": int(self._initial_pool.numel()),
            "remaining_pool_size": int(self._remaining_pool.numel()),
            "initial_samples_per_epoch": self.initial_samples_per_epoch,
            "remaining_samples_per_epoch": self.remaining_samples_per_epoch,
            "stream_samples_per_epoch": self.total_size,
            "padded_samples_per_epoch": self.total_size,
            "global_batches_per_epoch": self.global_batches_per_epoch,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "selection_policy": (
                "nearest_integer_10pct_episode_range_starts_"
                "90pct_all_other_indices_balanced_permutation_cycles"
            ),
            "shuffle_policy": "global_randperm_after_stratified_selection",
            "tail_policy": (
                "stream_length_matches_strict_padded_epoch_without_extra_tail"
            ),
        }

    def _epoch_indices(self) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.effective_epoch)
        initial = self._balanced_pool_sample(
            self._initial_pool,
            self.initial_samples_per_epoch,
            generator=generator,
        )
        remaining = self._balanced_pool_sample(
            self._remaining_pool,
            self.remaining_samples_per_epoch,
            generator=generator,
        )
        selected = torch.cat((initial, remaining))
        permutation = torch.randperm(
            self.total_size,
            generator=generator,
        )
        return selected[permutation].tolist()


def _normalize_failure_replay_manifest(value: object) -> dict[str, Any]:
    """Return one canonical, fail-closed failure-task manifest."""

    if not isinstance(value, Mapping):
        raise TypeError("failure replay manifest must be a mapping")
    required = {"schema_version", "split", "failures"}
    if set(value) != required:
        raise ValueError(
            "failure replay manifest fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )
    if value["schema_version"] != FAILURE_REPLAY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("failure replay manifest schema is unsupported")
    if value["split"] != "train":
        raise ValueError("failure replay manifest must bind split='train'")
    raw_failures = value["failures"]
    if not isinstance(raw_failures, (tuple, list)) or not raw_failures:
        raise ValueError("failure replay manifest failures must be non-empty")

    allowed = {"task_name", "variant", "task_config"}
    normalized: list[dict[str, str]] = []
    for position, raw in enumerate(raw_failures):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"failure replay selector {position} must be a mapping"
            )
        if "task_name" not in raw or not set(raw).issubset(allowed):
            raise ValueError(
                f"failure replay selector {position} fields are invalid"
            )
        selector: dict[str, str] = {}
        for field in ("task_name", "variant", "task_config"):
            if field not in raw:
                continue
            field_value = raw[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"failure replay selector {position} {field} "
                    "must be non-empty text"
                )
            selector[field] = field_value.strip()
        normalized.append(selector)

    normalized.sort(
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    serialized = [
        json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in normalized
    ]
    if len(serialized) != len(set(serialized)):
        raise ValueError("failure replay manifest contains duplicate selectors")
    return {
        "schema_version": FAILURE_REPLAY_MANIFEST_SCHEMA_VERSION,
        "split": "train",
        "failures": normalized,
    }


def _normalize_failure_replay_task_variant_noinitial_manifest(
    value: object,
) -> dict[str, Any]:
    """Return a canonical, fail-closed task×variant replay manifest.

    ``task_config`` is intentionally retained as a data-contract guard even
    though the sampling identity is task×variant.  RoboTwin's formal train
    index currently maps clean/randomized to demo_clean/demo_randomized; an
    accidental future remapping must fail rather than silently blend pools.
    """

    if not isinstance(value, Mapping):
        raise TypeError("task-variant replay manifest must be a mapping")
    required = {"schema_version", "split", "failure_cells"}
    if set(value) != required:
        raise ValueError(
            "task-variant replay manifest fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )
    if (
        value["schema_version"]
        != FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("task-variant replay manifest schema is unsupported")
    if value["split"] != "train":
        raise ValueError("task-variant replay manifest must bind split='train'")
    raw_cells = value["failure_cells"]
    if not isinstance(raw_cells, (tuple, list)) or not raw_cells:
        raise ValueError(
            "task-variant replay manifest failure_cells must be non-empty"
        )

    required_cell_fields = {"task_name", "variant", "task_config"}
    normalized: list[dict[str, str]] = []
    for position, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping) or set(raw) != required_cell_fields:
            raise ValueError(
                "task-variant replay cell "
                f"{position} must contain exactly "
                "task_name, variant, and task_config"
            )
        cell: dict[str, str] = {}
        for field in ("task_name", "variant", "task_config"):
            field_value = raw[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    "task-variant replay cell "
                    f"{position} {field} must be non-empty text"
                )
            cell[field] = field_value.strip()
        normalized.append(cell)

    normalized.sort(
        key=lambda item: (
            item["task_name"],
            item["variant"],
            item["task_config"],
        )
    )
    task_variant_keys = [
        (item["task_name"], item["variant"])
        for item in normalized
    ]
    if len(task_variant_keys) != len(set(task_variant_keys)):
        raise ValueError(
            "task-variant replay manifest contains duplicate task×variant "
            "cells"
        )
    return {
        "schema_version": (
            FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_MANIFEST_SCHEMA_VERSION
        ),
        "split": "train",
        "failure_cells": normalized,
    }


def canonical_task_variant_failure_manifest(
    value: object,
) -> dict[str, Any]:
    """Expose the sampler's exact manifest normalization contract."""

    return _normalize_failure_replay_task_variant_noinitial_manifest(value)


def canonical_task_variant_failure_manifest_sha256(value: object) -> str:
    """Hash a task×variant manifest exactly as the sampler binding does."""

    normalized = canonical_task_variant_failure_manifest(value)
    return _canonical_sha256(
        normalized,
        prefix=(
            b"fastwam-failure-replay-task-variant-noinitial-manifest-v1"
        ),
    )


def _ordered_tensor_pools_sha256(
    pools: tuple[tuple[str, torch.Tensor], ...],
    *,
    prefix: bytes,
) -> str:
    """Hash ordered int64 index pools without materializing JSON arrays."""

    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(b"\0")
    for name, pool in pools:
        if pool.dtype != torch.long or pool.ndim != 1:
            raise TypeError("sampler identity pools must be one-dimensional int64")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(int(pool.numel()).to_bytes(8, "little"))
        array = pool.detach().cpu().contiguous().numpy()
        digest.update(array.astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


class FailureReplayInitialWindowSampler(InitialWindowPreferredSampler):
    """Mix 80% full-train sampling with 20% task-balanced failure replay.

    Both the base and replay streams independently reserve 10% of their
    samples for the first legal window of an episode.  Replay first chooses a
    failed task uniformly (balanced permutation cycles), then chooses a legal
    window within that task's pool.  The final mixed stream is globally
    shuffled and has exactly the same EGB-padded epoch length as the strict
    all-window sampler.
    """

    _BASE_NUMERATOR = 4
    _REPLAY_NUMERATOR = 1
    _MIX_DENOMINATOR = 5

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
        *,
        failure_manifest: object,
        dataset_split_name: str,
    ) -> None:
        if dataset_split_name != "train":
            raise ValueError(
                "failure replay sampling is restricted to the train split"
            )
        self._dataset_split_name = dataset_split_name
        self._failure_manifest = _normalize_failure_replay_manifest(
            failure_manifest
        )
        super().__init__(
            dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        metadata = self._load_episode_replay_metadata(dataset)
        if len(metadata) != len(self._episode_ranges):
            raise ValueError(
                "train-cache episode metadata count differs from episode ranges"
            )
        selectors = tuple(self._failure_manifest["failures"])
        selectors_by_task: dict[str, list[tuple[int, Mapping[str, str]]]] = {}
        for position, selector in enumerate(selectors):
            selectors_by_task.setdefault(selector["task_name"], []).append(
                (position, selector)
            )

        selector_match_counts = [0] * len(selectors)
        initial_by_task: dict[str, list[int]] = {
            task_name: [] for task_name in selectors_by_task
        }
        remaining_by_task: dict[str, list[torch.Tensor]] = {
            task_name: [] for task_name in selectors_by_task
        }
        for (start, stop), episode in zip(self._episode_ranges, metadata):
            task_name = episode["task_name"]
            task_selectors = selectors_by_task.get(task_name, ())
            matched = False
            for selector_position, selector in task_selectors:
                if all(
                    episode.get(field) == expected
                    for field, expected in selector.items()
                    if field != "task_name"
                ):
                    selector_match_counts[selector_position] += 1
                    matched = True
            if not matched:
                continue
            initial_by_task[task_name].append(start)
            if stop > start + 1:
                remaining_by_task[task_name].append(
                    torch.arange(start + 1, stop, dtype=torch.long)
                )

        unmatched = [
            dict(selectors[position])
            for position, count in enumerate(selector_match_counts)
            if count == 0
        ]
        if unmatched:
            raise ValueError(
                "failure replay selectors match no train-cache episodes: "
                f"{unmatched}"
            )

        task_initial_pools: list[tuple[str, torch.Tensor]] = []
        task_remaining_pools: list[tuple[str, torch.Tensor]] = []
        task_pool_counts: list[dict[str, Any]] = []
        for task_name in sorted(selectors_by_task):
            initial = torch.tensor(
                initial_by_task[task_name], dtype=torch.long
            )
            remaining_chunks = remaining_by_task[task_name]
            remaining = (
                torch.cat(remaining_chunks)
                if remaining_chunks
                else torch.empty(0, dtype=torch.long)
            )
            if initial.numel() == 0 or remaining.numel() == 0:
                raise ValueError(
                    "every replay task must provide both initial and "
                    f"non-initial train windows: {task_name!r}"
                )
            task_initial_pools.append((task_name, initial))
            task_remaining_pools.append((task_name, remaining))
            task_pool_counts.append(
                {
                    "task_name": task_name,
                    "initial_pool_size": int(initial.numel()),
                    "remaining_pool_size": int(remaining.numel()),
                    "total_pool_size": int(initial.numel() + remaining.numel()),
                }
            )
        if not task_initial_pools:
            raise ValueError("failure replay task pool is empty")

        self._replay_initial_pools = tuple(task_initial_pools)
        self._replay_remaining_pools = tuple(task_remaining_pools)
        self._replay_task_pool_counts = tuple(task_pool_counts)
        self._failure_manifest_sha256 = _canonical_sha256(
            self._failure_manifest,
            prefix=b"fastwam-failure-replay-manifest-v1",
        )
        self._base_pool_sha256 = _ordered_tensor_pools_sha256(
            (
                ("initial", self._initial_pool),
                ("remaining", self._remaining_pool),
            ),
            prefix=b"fastwam-failure-replay-base-pools-int64le-v1",
        )
        self._replay_pool_sha256 = _ordered_tensor_pools_sha256(
            tuple(
                (f"initial:{task_name}", pool)
                for task_name, pool in self._replay_initial_pools
            )
            + tuple(
                (f"remaining:{task_name}", pool)
                for task_name, pool in self._replay_remaining_pools
            ),
            prefix=b"fastwam-failure-replay-task-pools-int64le-v1",
        )

        if min(
            self.base_initial_samples_per_epoch,
            self.base_remaining_samples_per_epoch,
            self.replay_initial_samples_per_epoch,
            self.replay_remaining_samples_per_epoch,
        ) <= 0:
            raise ValueError(
                "failure replay epoch is too small to realize both 80/20 "
                "and nested 10/90 strata"
            )

    @staticmethod
    def _load_episode_replay_metadata(
        dataset: Sized,
    ) -> tuple[dict[str, str | None], ...]:
        provider = getattr(dataset, "episode_replay_metadata", None)
        if callable(provider):
            raw_records = provider()
        else:
            source = getattr(dataset, "source", None)
            index = getattr(source, "index", None)
            raw_records = getattr(index, "episodes", None)
            if raw_records is None:
                sample_refs = getattr(dataset, "sample_refs", None)
                plans = getattr(sample_refs, "plans", None)
                if plans is not None:
                    raw_records = tuple(plan.episode for plan in plans)
        if not isinstance(raw_records, (tuple, list)) or not raw_records:
            raise TypeError(
                "failure replay requires ordered per-episode train-cache metadata"
            )

        result: list[dict[str, str | None]] = []
        for position, raw in enumerate(raw_records):
            def field(name: str) -> object:
                if isinstance(raw, Mapping):
                    return raw.get(name)
                return getattr(raw, name, None)

            task_name = field("task_name")
            if not isinstance(task_name, str) or not task_name.strip():
                raise ValueError(
                    "train-cache episode metadata has an invalid task_name: "
                    f"episode position {position}"
                )
            record: dict[str, str | None] = {
                "task_name": task_name.strip(),
            }
            for name in ("variant", "task_config"):
                value = field(name)
                if value is None:
                    record[name] = None
                elif not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "train-cache episode metadata has an invalid "
                        f"{name}: episode position {position}"
                    )
                else:
                    record[name] = value.strip()
            result.append(record)
        return tuple(result)

    @staticmethod
    def _rounded_fraction(total: int, numerator: int, denominator: int) -> int:
        return (total * numerator + denominator // 2) // denominator

    @property
    def replay_samples_per_epoch(self) -> int:
        return self._rounded_fraction(
            self.total_size,
            self._REPLAY_NUMERATOR,
            self._MIX_DENOMINATOR,
        )

    @property
    def base_samples_per_epoch(self) -> int:
        return self.total_size - self.replay_samples_per_epoch

    @property
    def base_initial_samples_per_epoch(self) -> int:
        return self._rounded_fraction(
            self.base_samples_per_epoch,
            self._INITIAL_NUMERATOR,
            self._INITIAL_DENOMINATOR,
        )

    @property
    def base_remaining_samples_per_epoch(self) -> int:
        return self.base_samples_per_epoch - self.base_initial_samples_per_epoch

    @property
    def replay_initial_samples_per_epoch(self) -> int:
        return self._rounded_fraction(
            self.replay_samples_per_epoch,
            self._INITIAL_NUMERATOR,
            self._INITIAL_DENOMINATOR,
        )

    @property
    def replay_remaining_samples_per_epoch(self) -> int:
        return (
            self.replay_samples_per_epoch
            - self.replay_initial_samples_per_epoch
        )

    @property
    def initial_samples_per_epoch(self) -> int:
        return (
            self.base_initial_samples_per_epoch
            + self.replay_initial_samples_per_epoch
        )

    @property
    def remaining_samples_per_epoch(self) -> int:
        return self.total_size - self.initial_samples_per_epoch

    @classmethod
    def _task_balanced_pool_sample(
        cls,
        pools: tuple[tuple[str, torch.Tensor], ...],
        count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        task_assignments = cls._balanced_pool_sample(
            torch.arange(len(pools), dtype=torch.long),
            count,
            generator=generator,
        )
        selected = torch.empty(count, dtype=torch.long)
        for task_position, (_, pool) in enumerate(pools):
            output_positions = torch.nonzero(
                task_assignments == task_position,
                as_tuple=False,
            ).flatten()
            selected[output_positions] = cls._balanced_pool_sample(
                pool,
                int(output_positions.numel()),
                generator=generator,
            )
        return selected

    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": FAILURE_REPLAY_SAMPLER_SCHEMA_VERSION,
            "sampling_mode": FAILURE_REPLAY_80PCT_INITIAL10,
            "dataset_split_name": self._dataset_split_name,
            "dataset_length": len(self.dataset),
            "episode_count": len(self._episode_ranges),
            "episode_ranges": [
                [int(start), int(stop)]
                for start, stop in self._episode_ranges
            ],
            "episode_ranges_binding_schema_version": str(
                self._episode_ranges_binding["schema_version"]
            ),
            "episode_ranges_digest_algorithm": str(
                self._episode_ranges_binding["digest_algorithm"]
            ),
            "episode_ranges_identity_sha256": str(
                self._episode_ranges_binding["ranges_sha256"]
            ),
            "episode_range_boundaries_sha256": (
                self._episode_range_boundaries_sha256
            ),
            "seed": self.seed,
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_microbatch_size": self.global_microbatch_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "base_fraction_numerator": self._BASE_NUMERATOR,
            "replay_fraction_numerator": self._REPLAY_NUMERATOR,
            "mix_fraction_denominator": self._MIX_DENOMINATOR,
            "initial_fraction_numerator": self._INITIAL_NUMERATOR,
            "initial_fraction_denominator": self._INITIAL_DENOMINATOR,
            "base_initial_pool_size": int(self._initial_pool.numel()),
            "base_remaining_pool_size": int(self._remaining_pool.numel()),
            "replay_task_count": len(self._replay_initial_pools),
            "replay_initial_pool_size": sum(
                int(pool.numel()) for _, pool in self._replay_initial_pools
            ),
            "replay_remaining_pool_size": sum(
                int(pool.numel()) for _, pool in self._replay_remaining_pools
            ),
            "replay_task_pool_counts": [
                dict(item) for item in self._replay_task_pool_counts
            ],
            "base_samples_per_epoch": self.base_samples_per_epoch,
            "replay_samples_per_epoch": self.replay_samples_per_epoch,
            "base_initial_samples_per_epoch": (
                self.base_initial_samples_per_epoch
            ),
            "base_remaining_samples_per_epoch": (
                self.base_remaining_samples_per_epoch
            ),
            "replay_initial_samples_per_epoch": (
                self.replay_initial_samples_per_epoch
            ),
            "replay_remaining_samples_per_epoch": (
                self.replay_remaining_samples_per_epoch
            ),
            "initial_samples_per_epoch": self.initial_samples_per_epoch,
            "remaining_samples_per_epoch": self.remaining_samples_per_epoch,
            "stream_samples_per_epoch": self.total_size,
            "selected_samples_per_epoch": self.total_size,
            "padded_samples_per_epoch": self.total_size,
            "global_batches_per_epoch": self.global_batches_per_epoch,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "failure_manifest": self._failure_manifest,
            "failure_manifest_sha256": self._failure_manifest_sha256,
            "pool_digest_algorithm": "ordered_named_int64le_sha256_v1",
            "base_pool_sha256": self._base_pool_sha256,
            "replay_pool_sha256": self._replay_pool_sha256,
            "selection_policy": (
                "80pct_full_train_20pct_failed_task_balanced_replay_"
                "with_nested_10pct_initial_90pct_remaining"
            ),
            "shuffle_policy": "global_randperm_after_four_strata_selection",
            "tail_policy": (
                "stream_length_matches_strict_padded_epoch_without_extra_tail"
            ),
        }

    def _epoch_indices(self) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.effective_epoch)
        base_initial = self._balanced_pool_sample(
            self._initial_pool,
            self.base_initial_samples_per_epoch,
            generator=generator,
        )
        base_remaining = self._balanced_pool_sample(
            self._remaining_pool,
            self.base_remaining_samples_per_epoch,
            generator=generator,
        )
        replay_initial = self._task_balanced_pool_sample(
            self._replay_initial_pools,
            self.replay_initial_samples_per_epoch,
            generator=generator,
        )
        replay_remaining = self._task_balanced_pool_sample(
            self._replay_remaining_pools,
            self.replay_remaining_samples_per_epoch,
            generator=generator,
        )
        selected = torch.cat(
            (base_initial, base_remaining, replay_initial, replay_remaining)
        )
        if int(selected.numel()) != self.total_size:
            raise AssertionError("failure replay stratum sizes do not sum to epoch")
        permutation = torch.randperm(self.total_size, generator=generator)
        return selected[permutation].tolist()


class FailureReplayTaskVariantNoInitialSampler(ResumableEpisodeWindowSampler):
    """Mix all legal windows with independent failed task×variant pools.

    This is deliberately *not* an ``InitialWindowPreferredSampler``.  The
    base 80% and replay 20% both draw from every legal window in their pool;
    episode starts may occur only at their natural pool frequency.  Replay is
    split equally between (a) failed tasks sampled uniformly, then one of
    that task's failed variants uniformly, and (b) failure cells sampled
    uniformly.  That gives persistent two-variant failures additional weight
    without letting either the randomized data volume or a task's number of
    failed variants silently dominate the replay stream.
    """

    _BASE_NUMERATOR = 4
    _REPLAY_NUMERATOR = 1
    _MIX_DENOMINATOR = 5
    _TASK_COMPONENT_NUMERATOR = 1
    _TASK_COMPONENT_DENOMINATOR = 2

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
        *,
        failure_manifest: object,
        dataset_split_name: str,
    ) -> None:
        if dataset_split_name != "train":
            raise ValueError(
                "task-variant failure replay sampling is restricted to the "
                "train split"
            )
        self._dataset_split_name = dataset_split_name
        self._failure_manifest = (
            _normalize_failure_replay_task_variant_noinitial_manifest(
                failure_manifest
            )
        )
        super().__init__(
            dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        metadata = FailureReplayInitialWindowSampler._load_episode_replay_metadata(
            dataset
        )
        if len(metadata) != len(self._episode_ranges):
            raise ValueError(
                "train-cache episode metadata count differs from episode ranges"
            )

        selectors = tuple(self._failure_manifest["failure_cells"])
        selectors_by_cell = {
            (selector["task_name"], selector["variant"]): selector
            for selector in selectors
        }
        if len(selectors_by_cell) != len(selectors):
            raise AssertionError(
                "canonical task-variant manifest unexpectedly duplicated a cell"
            )
        ordered_cells = tuple(sorted(selectors_by_cell))
        selector_match_counts = {cell: 0 for cell in ordered_cells}
        windows_by_cell: dict[tuple[str, str], list[torch.Tensor]] = {
            cell: [] for cell in ordered_cells
        }
        for (start, stop), episode in zip(self._episode_ranges, metadata):
            task_name = episode["task_name"]
            variant = episode.get("variant")
            if variant is None:
                continue
            cell = (task_name, variant)
            selector = selectors_by_cell.get(cell)
            if selector is None:
                continue
            if episode.get("task_config") != selector["task_config"]:
                continue
            selector_match_counts[cell] += 1
            windows_by_cell[cell].append(
                torch.arange(start, stop, dtype=torch.long)
            )

        unmatched = [
            dict(selectors_by_cell[cell])
            for cell in ordered_cells
            if selector_match_counts[cell] == 0
        ]
        if unmatched:
            raise ValueError(
                "task-variant replay cells match no train-cache episodes: "
                f"{unmatched}"
            )

        cell_pools: list[tuple[tuple[str, str], torch.Tensor]] = []
        cell_pool_counts: list[dict[str, Any]] = []
        for cell in ordered_cells:
            chunks = windows_by_cell[cell]
            pool = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)
            if pool.numel() == 0:
                raise ValueError(
                    "task-variant replay cell has no legal train windows: "
                    f"{dict(selectors_by_cell[cell])}"
                )
            cell_pools.append((cell, pool))
            cell_pool_counts.append(
                {
                    "task_name": cell[0],
                    "variant": cell[1],
                    "task_config": selectors_by_cell[cell]["task_config"],
                    "episode_count": selector_match_counts[cell],
                    "total_pool_size": int(pool.numel()),
                }
            )
        if not cell_pools:
            raise AssertionError("non-empty manifest unexpectedly made no pools")

        cells_by_task: dict[str, list[int]] = {}
        for cell_position, (cell, _) in enumerate(cell_pools):
            cells_by_task.setdefault(cell[0], []).append(cell_position)
        task_cell_positions = tuple(
            (task_name, tuple(cells_by_task[task_name]))
            for task_name in sorted(cells_by_task)
        )
        task_pool_counts = [
            {
                "task_name": task_name,
                "failure_cell_count": len(cell_positions),
                "total_pool_size": sum(
                    int(cell_pools[position][1].numel())
                    for position in cell_positions
                ),
            }
            for task_name, cell_positions in task_cell_positions
        ]

        self._base_pool = torch.arange(len(self.dataset), dtype=torch.long)
        self._replay_cell_pools = tuple(cell_pools)
        self._replay_task_cell_positions = task_cell_positions
        self._replay_cell_pool_counts = tuple(cell_pool_counts)
        self._replay_task_pool_counts = tuple(task_pool_counts)
        self._failure_manifest_sha256 = _canonical_sha256(
            self._failure_manifest,
            prefix=b"fastwam-failure-replay-task-variant-noinitial-manifest-v1",
        )
        self._base_pool_sha256 = _ordered_tensor_pools_sha256(
            (("all_legal_train_windows", self._base_pool),),
            prefix=b"fastwam-failure-replay-task-variant-noinitial-base-pool-v1",
        )
        self._replay_pool_sha256 = _ordered_tensor_pools_sha256(
            tuple(
                (
                    "cell:"
                    + json.dumps(
                        {
                            "task_name": cell[0],
                            "variant": cell[1],
                            "task_config": selectors_by_cell[cell]["task_config"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    pool,
                )
                for cell, pool in self._replay_cell_pools
            ),
            prefix=b"fastwam-failure-replay-task-variant-noinitial-pools-v1",
        )

        if min(
            self.base_samples_per_epoch,
            self.replay_samples_per_epoch,
            self.replay_task_samples_per_epoch,
            self.replay_cell_samples_per_epoch,
        ) <= 0:
            raise ValueError(
                "task-variant failure replay epoch is too small to realize "
                "80/20 and 50/50 replay components"
            )

    @property
    def total_size(self) -> int:
        return (
            ceil(len(self.dataset) / self.effective_global_batch_size)
            * self.effective_global_batch_size
        )

    @property
    def selected_samples_per_epoch(self) -> int:
        return self.total_size

    @staticmethod
    def _rounded_fraction(total: int, numerator: int, denominator: int) -> int:
        return (total * numerator + denominator // 2) // denominator

    @staticmethod
    def _balanced_pool_sample(
        pool: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return InitialWindowPreferredSampler._balanced_pool_sample(
            pool,
            count,
            generator=generator,
        )

    @property
    def replay_samples_per_epoch(self) -> int:
        return self._rounded_fraction(
            self.total_size,
            self._REPLAY_NUMERATOR,
            self._MIX_DENOMINATOR,
        )

    @property
    def base_samples_per_epoch(self) -> int:
        return self.total_size - self.replay_samples_per_epoch

    @property
    def replay_task_samples_per_epoch(self) -> int:
        return self._rounded_fraction(
            self.replay_samples_per_epoch,
            self._TASK_COMPONENT_NUMERATOR,
            self._TASK_COMPONENT_DENOMINATOR,
        )

    @property
    def replay_cell_samples_per_epoch(self) -> int:
        return self.replay_samples_per_epoch - self.replay_task_samples_per_epoch

    def _cell_pool_sample(
        self,
        assignments: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        selected = torch.empty(int(assignments.numel()), dtype=torch.long)
        for cell_position, (_, pool) in enumerate(self._replay_cell_pools):
            output_positions = torch.nonzero(
                assignments == cell_position,
                as_tuple=False,
            ).flatten()
            selected[output_positions] = self._balanced_pool_sample(
                pool,
                int(output_positions.numel()),
                generator=generator,
            )
        return selected

    def _task_component_sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        task_assignments = self._balanced_pool_sample(
            torch.arange(
                len(self._replay_task_cell_positions), dtype=torch.long
            ),
            count,
            generator=generator,
        )
        cell_assignments = torch.empty(count, dtype=torch.long)
        for task_position, (_, cell_positions) in enumerate(
            self._replay_task_cell_positions
        ):
            output_positions = torch.nonzero(
                task_assignments == task_position,
                as_tuple=False,
            ).flatten()
            cell_assignments[output_positions] = self._balanced_pool_sample(
                torch.tensor(cell_positions, dtype=torch.long),
                int(output_positions.numel()),
                generator=generator,
            )
        return self._cell_pool_sample(
            cell_assignments,
            generator=generator,
        )

    def _cell_component_sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assignments = self._balanced_pool_sample(
            torch.arange(len(self._replay_cell_pools), dtype=torch.long),
            count,
            generator=generator,
        )
        return self._cell_pool_sample(assignments, generator=generator)

    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": (
                FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_SAMPLER_SCHEMA_VERSION
            ),
            "sampling_mode": FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
            "dataset_split_name": self._dataset_split_name,
            "dataset_length": len(self.dataset),
            "episode_count": len(self._episode_ranges),
            "episode_ranges": [
                [int(start), int(stop)]
                for start, stop in self._episode_ranges
            ],
            "episode_ranges_binding_schema_version": str(
                self._episode_ranges_binding["schema_version"]
            ),
            "episode_ranges_digest_algorithm": str(
                self._episode_ranges_binding["digest_algorithm"]
            ),
            "episode_ranges_identity_sha256": str(
                self._episode_ranges_binding["ranges_sha256"]
            ),
            "episode_range_boundaries_sha256": (
                self._episode_range_boundaries_sha256
            ),
            "seed": self.seed,
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_microbatch_size": self.global_microbatch_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "base_fraction_numerator": self._BASE_NUMERATOR,
            "replay_fraction_numerator": self._REPLAY_NUMERATOR,
            "mix_fraction_denominator": self._MIX_DENOMINATOR,
            "replay_task_component_fraction_numerator": (
                self._TASK_COMPONENT_NUMERATOR
            ),
            "replay_task_component_fraction_denominator": (
                self._TASK_COMPONENT_DENOMINATOR
            ),
            "initial_window_fraction_numerator": 0,
            "initial_window_fraction_denominator": 1,
            "base_pool_size": int(self._base_pool.numel()),
            "replay_failure_task_count": len(
                self._replay_task_cell_positions
            ),
            "replay_failure_cell_count": len(self._replay_cell_pools),
            "replay_task_pool_counts": [
                dict(item) for item in self._replay_task_pool_counts
            ],
            "replay_cell_pool_counts": [
                dict(item) for item in self._replay_cell_pool_counts
            ],
            "base_samples_per_epoch": self.base_samples_per_epoch,
            "replay_samples_per_epoch": self.replay_samples_per_epoch,
            "replay_task_samples_per_epoch": (
                self.replay_task_samples_per_epoch
            ),
            "replay_cell_samples_per_epoch": (
                self.replay_cell_samples_per_epoch
            ),
            "stream_samples_per_epoch": self.total_size,
            "selected_samples_per_epoch": self.total_size,
            "padded_samples_per_epoch": self.total_size,
            "global_batches_per_epoch": self.global_batches_per_epoch,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "failure_manifest": self._failure_manifest,
            "failure_manifest_sha256": self._failure_manifest_sha256,
            "pool_digest_algorithm": "ordered_named_int64le_sha256_v1",
            "base_pool_sha256": self._base_pool_sha256,
            "replay_pool_sha256": self._replay_pool_sha256,
            "selection_policy": (
                "80pct_all_legal_train_windows_20pct_failed_task_variant_"
                "replay_half_task_uniform_then_variant_uniform_half_cell_uniform_"
                "no_initial_window_stratum"
            ),
            "shuffle_policy": "global_randperm_after_three_strata_selection",
            "tail_policy": (
                "stream_length_matches_strict_padded_epoch_without_extra_tail"
            ),
        }

    def build_manifest_replacement(
        self,
        failure_manifest: object,
    ) -> "FailureReplayTaskVariantNoInitialSampler":
        """Build and validate a replacement without mutating this sampler."""

        replacement = type(self)(
            self.dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.num_processes,
            gradient_accumulation_steps=(
                self.gradient_accumulation_steps
            ),
            failure_manifest=failure_manifest,
            dataset_split_name=self._dataset_split_name,
        )
        replacement.set_epoch(self.epoch)
        replacement.set_epoch_offset(self.epoch_offset)
        if self.resume_batch_offset:
            replacement.set_resume_batch_offset(
                self.resume_batch_offset
            )
        return replacement

    def adopt_manifest_replacement(
        self,
        replacement: "FailureReplayTaskVariantNoInitialSampler",
    ) -> None:
        """Atomically adopt only a prevalidated manifest-dependent pool set."""

        if not isinstance(
            replacement,
            FailureReplayTaskVariantNoInitialSampler,
        ):
            raise TypeError(
                "replacement must be a task-variant no-initial sampler"
            )
        current = self.binding()
        candidate = replacement.binding()
        mutable_fields = {
            "replay_failure_task_count",
            "replay_failure_cell_count",
            "replay_task_pool_counts",
            "replay_cell_pool_counts",
            "failure_manifest",
            "failure_manifest_sha256",
            "replay_pool_sha256",
        }
        current_static = {
            key: value
            for key, value in current.items()
            if key not in mutable_fields
        }
        candidate_static = {
            key: value
            for key, value in candidate.items()
            if key not in mutable_fields
        }
        if current_static != candidate_static:
            raise ValueError(
                "replacement changes immutable sampler math or topology"
            )
        for attribute in (
            "_failure_manifest",
            "_replay_cell_pools",
            "_replay_task_cell_positions",
            "_replay_cell_pool_counts",
            "_replay_task_pool_counts",
            "_failure_manifest_sha256",
            "_replay_pool_sha256",
        ):
            setattr(self, attribute, getattr(replacement, attribute))

    def _epoch_indices(self) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.effective_epoch)
        base = self._balanced_pool_sample(
            self._base_pool,
            self.base_samples_per_epoch,
            generator=generator,
        )
        replay_task = self._task_component_sample(
            self.replay_task_samples_per_epoch,
            generator=generator,
        )
        replay_cell = self._cell_component_sample(
            self.replay_cell_samples_per_epoch,
            generator=generator,
        )
        selected = torch.cat((base, replay_task, replay_cell))
        if int(selected.numel()) != self.total_size:
            raise AssertionError(
                "task-variant failure replay stratum sizes do not sum to epoch"
            )
        permutation = torch.randperm(self.total_size, generator=generator)
        return selected[permutation].tolist()
