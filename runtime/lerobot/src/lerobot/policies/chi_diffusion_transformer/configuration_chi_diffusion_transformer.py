#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("chi_diffusion_transformer")
@dataclass
class ChiDiffusionTransformerConfig(PreTrainedConfig):
    """Official-compatible Diffusion Policy Transformer configuration.

    This is a LeRobot port of the real-stanford Diffusion Policy
    `TransformerForDiffusion` code path: noisy action tokens are decoded with a
    PreNorm Transformer decoder conditioned on timestep and observation tokens
    through cross-attention.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int | None = None

    # Vision encoder settings match the classic LeRobot diffusion policy.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True

    # Transformer trunk.
    n_layer: int = 8
    n_cond_layers: int = 0
    n_head: int = 4
    n_emb: int = 256
    dim_feedforward_multiplier: int = 4
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3
    causal_attn: bool = True
    time_as_cond: bool = True
    obs_as_cond: bool = True
    use_official_transformer: bool = False
    attnres_mode: str = "none"
    attnres_placement: str = "block"  # block, branch, sublayer
    attnres_block_size: int = 4
    attnres_warm_start: bool = True
    attnres_warm_start_bias: float = 8.0
    attnres_query_init: str = "zero"
    attnres_query_init_std: float = 0.02
    attnres_return_weights: bool = False

    # Kimi sublayer-specific config
    attnres_use_rmsnorm: bool = True  # Use RMSNorm on key (Kimi official)
    attnres_route_before_self_attn: bool = True  # Route before self-attention
    attnres_route_before_cross_attn: bool = False  # Route before cross-attention (optional)
    attnres_route_before_ffn: bool = True  # Route before FFN/MLP

    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # Optimization.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-3
    optimizer_obs_encoder_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 10.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 1000
    use_ema: bool = False
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999

    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

        if self.resize_shape is not None:
            if len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape):
                raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None

        self._validate()

    def _validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not (0 < self.n_action_steps <= self.horizon - self.n_obs_steps + 1):
            raise ValueError(
                "n_action_steps must be in (0, horizon - n_obs_steps + 1]. "
                f"Got {self.n_action_steps=}, {self.horizon=}, {self.n_obs_steps=}."
            )
        if self.n_emb % self.n_head != 0:
            raise ValueError(f"n_emb must be divisible by n_head. Got {self.n_emb=} and {self.n_head=}.")
        if self.noise_scheduler_type not in ["DDPM", "DDIM"]:
            raise ValueError(f"Unsupported noise_scheduler_type: {self.noise_scheduler_type}")
        if self.prediction_type not in ["epsilon", "sample"]:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
        if not self.obs_as_cond:
            raise ValueError("Only obs_as_cond=True is currently supported for ChiDiffusionTransformerPolicy.")
        if not self.time_as_cond:
            raise ValueError("Only time_as_cond=True is currently supported for ChiDiffusionTransformerPolicy.")
        if self.use_official_transformer:
            if self.dim_feedforward_multiplier != 4:
                raise ValueError("use_official_transformer=True requires dim_feedforward_multiplier=4.")
            if self.attnres_mode != "none":
                raise ValueError("use_official_transformer=True is only valid for the no-AttnRes baseline.")
        if self.attnres_mode not in {"none", "full", "block", "uniform", "scalar"}:
            raise ValueError(
                "`attnres_mode` must be one of {'none', 'full', 'block', 'uniform', 'scalar'}. "
                f"Got {self.attnres_mode}."
            )
        if self.attnres_placement not in {"block", "branch", "sublayer", "faithful"}:
            raise ValueError(
                "`attnres_placement` must be one of {'block', 'branch', 'sublayer', 'faithful'}. "
                f"Got {self.attnres_placement}."
            )
        if self.attnres_block_size <= 0:
            raise ValueError(f"`attnres_block_size` must be positive. Got {self.attnres_block_size}.")
        if self.attnres_warm_start_bias < 0:
            raise ValueError(
                f"`attnres_warm_start_bias` must be non-negative. Got {self.attnres_warm_start_bias}."
            )
        if self.attnres_query_init not in {"zero", "normal"}:
            raise ValueError(
                "`attnres_query_init` must be one of {'zero', 'normal'}. "
                f"Got {self.attnres_query_init}."
            )
        if self.attnres_query_init_std <= 0:
            raise ValueError(
                f"`attnres_query_init_std` must be positive. Got {self.attnres_query_init_std}."
            )
        if self.ema_update_after_step < 0:
            raise ValueError(f"`ema_update_after_step` must be non-negative. Got {self.ema_update_after_step}.")
        if self.ema_inv_gamma <= 0:
            raise ValueError(f"`ema_inv_gamma` must be positive. Got {self.ema_inv_gamma}.")
        if self.ema_power <= 0:
            raise ValueError(f"`ema_power` must be positive. Got {self.ema_power}.")
        if not (0 <= self.ema_min_value <= self.ema_max_value <= 1):
            raise ValueError(
                "`ema_min_value` and `ema_max_value` must satisfy 0 <= min <= max <= 1. "
                f"Got {self.ema_min_value=} and {self.ema_max_value=}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            # Official DPT uses parameter-group weight decay. Keep the optimizer
            # default at zero so only explicit groups decay.
            weight_decay=0.0,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("ChiDiffusionTransformerPolicy requires `observation.state` as input.")
        if self.action_feature is None:
            raise ValueError("ChiDiffusionTransformerPolicy requires `action` as output.")
        if self.image_features:
            first_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_key}`, but all image shapes must match."
                    )
                effective_shape = self.resize_shape or image_ft.shape[1:]
                if self.crop_shape is not None and (
                    self.crop_shape[0] > effective_shape[0] or self.crop_shape[1] > effective_shape[1]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within image shapes. Got {self.crop_shape} for `{key}`."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
