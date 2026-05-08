from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from cs336_scaling.scaling_config import TrainingConfig


@dataclass(frozen=True)
class LocalTrainingApi:
    """Deterministic local surrogate for the assignment's hidden training API.

    The surrogate is intentionally simple and interpretable: final loss follows a
    Chinchilla-style separable law in model size N and training tokens D, with
    small penalties for suboptimal learning-rate and batch-size choices.
    """

    irreducible_loss: float = 2.2
    model_coefficient: float = 95.0
    data_coefficient: float = 160.0
    model_exponent: float = 0.34
    data_exponent: float = 0.28
    noise_scale: float = 0.004

    def loss(self, config: TrainingConfig) -> dict[str, float | int]:
        n_params = float(config.non_embedding_params)
        d_tokens = float(config.train_flops) / (6.0 * n_params)

        scaling_loss = (
            self.irreducible_loss
            + self.model_coefficient / n_params**self.model_exponent
            + self.data_coefficient / d_tokens**self.data_exponent
        )
        loss = scaling_loss + self._learning_rate_penalty(config, n_params)
        loss += self._batch_size_penalty(config, n_params)
        loss += self._shape_penalty(config)
        loss += self._deterministic_noise(config)

        return {
            **config.to_record(),
            "dataset_tokens": d_tokens,
            "loss": float(loss),
        }

    def _learning_rate_penalty(self, config: TrainingConfig, n_params: float) -> float:
        # Larger models prefer slightly smaller learning rates.
        lr_opt = 6e-4 * (n_params / 1e7) ** -0.08
        lr_opt = min(1e-3, max(1e-4, lr_opt))
        log_error = math.log(config.learning_rate / lr_opt)
        return 0.09 * log_error * log_error

    def _batch_size_penalty(self, config: TrainingConfig, n_params: float) -> float:
        preferred_batch_size = 256 if n_params >= 6e7 else 128
        return 0.025 if config.batch_size != preferred_batch_size else 0.0

    def _shape_penalty(self, config: TrainingConfig) -> float:
        head_dim = config.d_model / config.num_heads
        return 0.01 * (math.log(head_dim / 64.0) ** 2)

    def _deterministic_noise(self, config: TrainingConfig) -> float:
        key = repr(config.key()).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return self.noise_scale * (2.0 * value - 1.0)
