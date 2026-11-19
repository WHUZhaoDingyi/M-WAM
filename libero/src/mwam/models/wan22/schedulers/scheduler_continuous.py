import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample


    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta


class WanHighNoise405505UniformWeightFlowMatchScheduler(
    WanContinuousFlowMatchScheduler
):
    """Opt-in 40/55/5 high-noise training policy with uniform loss weight.

    The two continuous buckets retain the base scheduler's shifted-uniform
    density after conditioning it on ``sigma < 0.95`` and
    ``0.95 <= sigma < 1`` respectively.  The final bucket is the exact
    ``sigma == 1`` endpoint.  Inference, interpolation, target velocity, and
    Euler stepping continue to use the base scheduler implementation.
    """

    TRAINING_POLICY_SCHEMA_VERSION = (
        "wan_high_noise_405505_uniform_weight_v1"
    )
    SIGMA_BOUNDARY = 0.95
    LOW_PROBABILITY = 0.40
    HIGH_PROBABILITY = 0.55
    ENDPOINT_PROBABILITY = 0.05
    ENDPOINT_SIGMA = 1.0

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            eps=eps,
        )
        denominator = self.shift - (
            (self.shift - 1.0) * self.SIGMA_BOUNDARY
        )
        if denominator <= 0.0:
            raise ValueError(
                "High-noise scheduler sigma boundary has no finite shifted-"
                "uniform inverse for the configured shift."
            )
        self._u_boundary = self.SIGMA_BOUNDARY / denominator
        if not 0.0 < self._u_boundary < 1.0:
            raise ValueError(
                "High-noise scheduler inverse boundary must lie in (0,1)."
            )

    def training_policy_contract(self) -> dict[str, object]:
        """Return the finite policy identity for checkpoint/audit binding."""

        return {
            "schema_version": self.TRAINING_POLICY_SCHEMA_VERSION,
            "continuous_density": (
                "base_shifted_uniform_conditioned_per_interval_v1"
            ),
            "sigma_boundary": self.SIGMA_BOUNDARY,
            "low_interval": "[0,0.95)",
            "high_interval": "[0.95,1)",
            "endpoint_sigma": self.ENDPOINT_SIGMA,
            "low_probability": self.LOW_PROBABILITY,
            "high_probability": self.HIGH_PROBABILITY,
            "endpoint_probability": self.ENDPOINT_PROBABILITY,
            "timestep_weighting": "uniform_one_v1",
        }

    def _timesteps_from_uniform_draws(
        self,
        bucket_draw: torch.Tensor,
        within_draw: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map two independent U[0,1) draws to the frozen training policy."""

        for value, name in (
            (bucket_draw, "bucket_draw"),
            (within_draw, "within_draw"),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`{name}` must be a torch.Tensor.")
            if value.ndim != 1:
                raise ValueError(f"`{name}` must be one-dimensional.")
            if not value.is_floating_point():
                raise TypeError(f"`{name}` must be floating point.")
        if bucket_draw.shape != within_draw.shape:
            raise ValueError("Uniform training draws must have identical shapes.")
        if bucket_draw.device != within_draw.device:
            raise ValueError("Uniform training draws must share one device.")

        bucket = bucket_draw.to(dtype=torch.float32)
        within = within_draw.to(dtype=torch.float32)
        low = bucket < self.LOW_PROBABILITY
        high = (bucket >= self.LOW_PROBABILITY) & (
            bucket < self.LOW_PROBABILITY + self.HIGH_PROBABILITY
        )

        u_boundary = torch.as_tensor(
            self._u_boundary,
            device=bucket.device,
            dtype=torch.float32,
        )
        low_u = within * u_boundary
        high_u = u_boundary + within * (1.0 - u_boundary)
        continuous_u = torch.where(low, low_u, high_u)
        continuous_sigma = self._phi(continuous_u, self.shift)

        steps = torch.as_tensor(
            float(self.num_train_timesteps),
            device=bucket.device,
            dtype=torch.float32,
        )
        zero = torch.zeros((), device=bucket.device, dtype=torch.float32)
        boundary_t = torch.as_tensor(
            self.SIGMA_BOUNDARY * float(self.num_train_timesteps),
            device=bucket.device,
            dtype=torch.float32,
        )
        low_upper = torch.nextafter(boundary_t, zero)
        continuous_upper = torch.nextafter(steps, zero)
        continuous_t = continuous_sigma * steps
        low_t = torch.minimum(continuous_t, low_upper)
        high_t = torch.clamp(
            continuous_t,
            min=boundary_t,
            max=continuous_upper,
        )
        timestep_fp32 = torch.where(
            low,
            low_t,
            torch.where(high, high_t, steps),
        )

        timestep = timestep_fp32.to(dtype=dtype)
        # Preserve an exact, auditable endpoint bucket even when a caller asks
        # for a lower-precision floating dtype. PixelDiT training requests FP32.
        if timestep.is_floating_point():
            output_steps = torch.as_tensor(
                float(self.num_train_timesteps),
                device=timestep.device,
                dtype=timestep.dtype,
            )
            output_zero = torch.zeros(
                (),
                device=timestep.device,
                dtype=timestep.dtype,
            )
            output_continuous_upper = torch.nextafter(
                output_steps,
                output_zero,
            )
            timestep = torch.where(
                low | high,
                torch.minimum(timestep, output_continuous_upper),
                output_steps,
            )
        return timestep

    def sample_training_t(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        draws = torch.rand(
            (2, batch_size),
            device=device,
            dtype=torch.float32,
        )
        return self._timesteps_from_uniform_draws(
            draws[0],
            draws[1],
            dtype=dtype,
        )

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            raise TypeError("`timestep` must be a torch.Tensor.")
        weight = torch.ones_like(timestep, dtype=torch.float32)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight


class WanUniformSigmaUniformWeightFlowMatchScheduler(WanContinuousFlowMatchScheduler):
    """Uniform sigma control; inference schedule and loss weight stay matched."""
    def sample_training_t(self, batch_size, device, dtype):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        return (torch.rand(batch_size,device=device,dtype=torch.float32)*self.num_train_timesteps).to(dtype)

    def training_weight(self,timestep):
        result=torch.ones_like(timestep,dtype=torch.float32)
        return result.reshape(()) if result.numel()==1 else result

    def training_policy_contract(self):
        return dict(schema_version='wan_uniform_sigma_uniform_weight_v1',
                    sigma_distribution='uniform_[0,1)',timestep_weighting='uniform_one_v1',
                    inference_shift=self.shift)
