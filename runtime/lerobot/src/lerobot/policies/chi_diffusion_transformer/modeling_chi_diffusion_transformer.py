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

import math
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _diffusers_available, require_package

from .attention_residuals import AttentionResiduals, FaithfulAttnRes, RMSNorm, SublayerAttnRes
from .configuration_chi_diffusion_transformer import ChiDiffusionTransformerConfig
from .official_transformer_for_diffusion import OfficialTransformerForDiffusion

if TYPE_CHECKING or _diffusers_available:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
else:
    DDIMScheduler = None
    DDPMScheduler = None


class ChiDiffusionTransformerPolicy(PreTrainedPolicy):
    """LeRobot port of the real-stanford Diffusion Policy Transformer policy."""

    config_class = ChiDiffusionTransformerConfig
    name = "chi_diffusion_transformer"

    def __init__(self, config: ChiDiffusionTransformerConfig, **kwargs):
        require_package("diffusers", extra="diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.diffusion = ChiDiffusionTransformerModel(config)
        self.ema_diffusion = None
        if config.use_ema:
            self.ema_diffusion = deepcopy(self.diffusion)
            self.ema_diffusion.requires_grad_(False)
        self.register_buffer("_ema_step", torch.tensor(0, dtype=torch.long), persistent=True)
        self.reset()

    def get_optim_params(self) -> list[dict]:
        return self.diffusion.get_optim_groups(
            transformer_weight_decay=self.config.optimizer_weight_decay,
            obs_encoder_weight_decay=self.config.optimizer_obs_encoder_weight_decay,
        )

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}
        expected = {key for key in (OBS_STATE, OBS_IMAGES, OBS_ENV_STATE) if key in self._queues}
        missing = expected - batch.keys()
        if missing:
            raise ValueError(f"Missing queued observation keys for Chi diffusion action prediction: {sorted(missing)}")
        return self._inference_diffusion().generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        loss = self.diffusion.compute_loss(batch)
        return loss, None

    def _inference_diffusion(self) -> "ChiDiffusionTransformerModel":
        if self.config.use_ema and not self.training:
            if self.ema_diffusion is None:
                raise RuntimeError("EMA inference requested but EMA weights are not initialized.")
            return self.ema_diffusion
        return self.diffusion

    def _ema_decay(self) -> float:
        step = int(self._ema_step.item())
        if step <= self.config.ema_update_after_step:
            return 0.0
        value = 1 - (1 + (step - self.config.ema_update_after_step) / self.config.ema_inv_gamma) ** (
            -self.config.ema_power
        )
        return min(max(value, self.config.ema_min_value), self.config.ema_max_value)

    @torch.no_grad()
    def update(self) -> None:
        if not self.config.use_ema:
            return
        if self.ema_diffusion is None:
            raise RuntimeError("EMA update requested but EMA weights are not initialized.")

        decay = self._ema_decay()
        model_params = dict(self.diffusion.named_parameters())
        ema_params = dict(self.ema_diffusion.named_parameters())
        for name, param in model_params.items():
            ema_param = ema_params[name]
            ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)

        model_buffers = dict(self.diffusion.named_buffers())
        ema_buffers = dict(self.ema_diffusion.named_buffers())
        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer)

        self._ema_step += 1


def _make_noise_scheduler(name: str, **kwargs: dict):
    require_package("diffusers", extra="diffusion")

    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {name}")


class ChiDiffusionTransformerModel(nn.Module):
    def __init__(self, config: ChiDiffusionTransformerConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_feature.shape[0]

        obs_token_dim = config.robot_state_feature.shape[0]
        if config.env_state_feature:
            obs_token_dim += config.env_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                obs_token_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                obs_token_dim += self.rgb_encoder.feature_dim * num_images

        if config.use_official_transformer:
            self.transformer = OfficialTransformerForDiffusion(
                input_dim=self.action_dim,
                output_dim=self.action_dim,
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=obs_token_dim,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_emb=config.n_emb,
                p_drop_emb=config.p_drop_emb,
                p_drop_attn=config.p_drop_attn,
                causal_attn=config.causal_attn,
                time_as_cond=config.time_as_cond,
                obs_as_cond=config.obs_as_cond,
                n_cond_layers=config.n_cond_layers,
            )
        else:
            self.transformer = TransformerForDiffusion(
                input_dim=self.action_dim,
                output_dim=self.action_dim,
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=obs_token_dim,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_emb=config.n_emb,
                dim_feedforward_multiplier=config.dim_feedforward_multiplier,
                p_drop_emb=config.p_drop_emb,
                p_drop_attn=config.p_drop_attn,
                causal_attn=config.causal_attn,
                n_cond_layers=config.n_cond_layers,
                attnres_mode=config.attnres_mode,
                attnres_placement=config.attnres_placement,
                attnres_block_size=config.attnres_block_size,
                attnres_warm_start=config.attnres_warm_start,
                attnres_warm_start_bias=config.attnres_warm_start_bias,
                attnres_query_init=config.attnres_query_init,
                attnres_query_init_std=config.attnres_query_init_std,
                attnres_return_weights=config.attnres_return_weights,
                attnres_use_rmsnorm=config.attnres_use_rmsnorm,
                attnres_route_before_self_attn=config.attnres_route_before_self_attn,
                attnres_route_before_cross_attn=config.attnres_route_before_cross_attn,
                attnres_route_before_ffn=config.attnres_route_before_ffn,
            )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    def get_optim_groups(
        self,
        transformer_weight_decay: float = 1e-3,
        obs_encoder_weight_decay: float = 1e-6,
    ) -> list[dict]:
        optim_groups = self.transformer.get_optim_groups(weight_decay=transformer_weight_decay)
        if self.config.image_features:
            optim_groups.append(
                {
                    "params": [p for p in self.rgb_encoder.parameters() if p.requires_grad],
                    "weight_decay": obs_encoder_weight_decay,
                }
            )
        return optim_groups

    def _prepare_obs_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        cond_features = [batch[OBS_STATE]]

        if self.config.env_state_feature:
            cond_features.append(batch[OBS_ENV_STATE])

        if self.config.image_features:
            images = batch[OBS_IMAGES]
            num_images = images.shape[2]
            if self.config.use_separate_rgb_encoder_per_camera:
                encoded_images = []
                for img_idx, encoder in enumerate(self.rgb_encoder):
                    img = images[:, :, img_idx].reshape(batch_size * n_obs_steps, *images.shape[3:])
                    encoded_images.append(encoder(img).reshape(batch_size, n_obs_steps, -1))
                cond_features.extend(encoded_images)
            else:
                img = images.reshape(batch_size * n_obs_steps * num_images, *images.shape[3:])
                encoded = self.rgb_encoder(img).reshape(batch_size, n_obs_steps, num_images, -1)
                cond_features.append(encoded.flatten(start_dim=2))

        return torch.cat(cond_features, dim=-1)

    def conditional_sample(
        self,
        batch_size: int,
        cond: Tensor,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.action_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.transformer(sample, timestep, cond)
            sample = self.noise_scheduler.step(model_output, timestep, sample, generator=generator).prev_sample

        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size = batch[OBS_STATE].shape[0]
        cond = self._prepare_obs_conditioning(batch)
        actions = self.conditional_sample(batch_size=batch_size, cond=cond, noise=noise)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        required = {OBS_STATE, ACTION}
        if not required.issubset(batch):
            raise ValueError(f"Batch must contain {required}. Got keys: {sorted(batch)}")
        if self.config.image_features and OBS_IMAGES not in batch:
            raise ValueError(f"Batch must contain {OBS_IMAGES} when image features are configured.")
        if self.config.env_state_feature and OBS_ENV_STATE not in batch:
            raise ValueError(f"Batch must contain {OBS_ENV_STATE} when env-state features are configured.")

        trajectory = batch[ACTION]
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        cond = self._prepare_obs_conditioning(batch)
        pred = self.transformer(noisy_trajectory, timesteps, cond)

        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("`action_is_pad` is required when do_mask_loss_for_padding=True.")
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)
        return loss.mean()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def _causal_mask(size: int) -> Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.float().masked_fill(mask == 1, float("-inf"))


def _memory_mask(target_size: int, source_size: int) -> Tensor:
    target_idx, source_idx = torch.meshgrid(torch.arange(target_size), torch.arange(source_size), indexing="ij")
    # Source token 0 is timestep. Observation source token k+1 is visible from action token k onward.
    visible = target_idx >= (source_idx - 1)
    return torch.zeros(target_size, source_size).masked_fill(~visible, float("-inf"))


class TransformerForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int,
        cond_dim: int,
        n_layer: int,
        n_head: int,
        n_emb: int,
        dim_feedforward_multiplier: int = 4,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
        attnres_mode: str = "none",
        attnres_placement: str = "block",
        attnres_block_size: int = 4,
        attnres_warm_start: bool = True,
        attnres_warm_start_bias: float = 8.0,
        attnres_query_init: str = "zero",
        attnres_query_init_std: float = 0.02,
        attnres_return_weights: bool = False,
        attnres_use_rmsnorm: bool = True,
        attnres_route_before_self_attn: bool = True,
        attnres_route_before_cross_attn: bool = False,
        attnres_route_before_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        cond_tokens = 1 + n_obs_steps
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, cond_tokens, n_emb))
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=dim_feedforward_multiplier * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=dim_feedforward_multiplier * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)
        self.attnres = None
        self.sublayer_attnres = None
        self.faithful_attnres = None
        self.attnres_mode = attnres_mode
        self.attnres_block_size = attnres_block_size
        self.attnres_placement = attnres_placement
        self.attnres_route_before_self_attn = attnres_route_before_self_attn
        self.attnres_route_before_cross_attn = attnres_route_before_cross_attn
        self.attnres_route_before_ffn = attnres_route_before_ffn

        if attnres_mode != "none":
            if attnres_placement not in {"block", "branch", "sublayer", "faithful"}:
                raise ValueError(
                    "`attnres_placement` must be one of {'block', 'branch', 'sublayer', 'faithful'}. "
                    f"Got {attnres_placement}."
                )
            if attnres_placement == "faithful":
                # Faithful Kimi AttnRes: one aggregation point per decoder layer
                # (layer-level, mode full|block). v_0 = decoder input embedding,
                # v_i = raw output of layer i. Independent query per point.
                if attnres_mode not in {"full", "block"}:
                    raise ValueError(
                        "`faithful` placement requires attnres_mode in {'full', 'block'}. "
                        f"Got {attnres_mode}."
                    )
                self.faithful_attnres = FaithfulAttnRes(
                    dim_model=n_emb,
                    num_units=n_layer,
                    mode=attnres_mode,
                    use_rmsnorm=attnres_use_rmsnorm,
                    query_init=attnres_query_init,
                    query_init_std=attnres_query_init_std,
                    return_weights=attnres_return_weights,
                )
            elif attnres_placement == "sublayer":
                self.sublayer_attnres = SublayerAttnRes(
                    dim_model=n_emb,
                    num_layers=n_layer,
                    block_size=attnres_block_size,
                    use_rmsnorm=attnres_use_rmsnorm,
                    query_init=attnres_query_init,
                    query_init_std=attnres_query_init_std,
                )
            else:
                num_attnres_units = n_layer if attnres_placement == "block" else n_layer * 3
                self.attnres = AttentionResiduals(
                    dim_model=n_emb,
                    num_layers=num_attnres_units,
                    mode=attnres_mode,
                    block_size=attnres_block_size,
                    warm_start=attnres_warm_start,
                    warm_start_bias=attnres_warm_start_bias,
                    query_init=attnres_query_init,
                    query_init_std=attnres_query_init_std,
                    return_weights=attnres_return_weights,
                )
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        if causal_attn:
            self.register_buffer("mask", _causal_mask(horizon), persistent=False)
            self.register_buffer("memory_mask", _memory_mask(horizon, cond_tokens), persistent=False)
        else:
            self.mask = None
            self.memory_mask = None

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]:
                weight = getattr(module, name, None)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ["in_proj_bias", "bias_k", "bias_v"]:
                bias = getattr(module, name, None)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_optim_groups(self, weight_decay: float = 1e-3) -> list[dict]:
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, RMSNorm)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias") or param_name.startswith("bias"):
                    no_decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif (
                    param_name.endswith("weight")
                    and isinstance(module, blacklist_weight_modules)
                    or full_param_name in {"pos_emb", "cond_pos_emb"}
                    or isinstance(module, (AttentionResiduals, SublayerAttnRes, FaithfulAttnRes))
                ):
                    no_decay.add(full_param_name)

        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        inter_params = decay & no_decay
        if inter_params:
            raise RuntimeError(f"Parameters in both decay/no_decay groups: {sorted(inter_params)}")
        missing_params = param_dict.keys() - (decay | no_decay)
        if missing_params:
            raise RuntimeError(f"Parameters missing from optimizer groups: {sorted(missing_params)}")

        return [
            {
                "params": [param_dict[name] for name in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[name] for name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def forward(self, sample: Tensor, timestep: Tensor | float | int, cond: Tensor) -> Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        time_emb = self.time_emb(timestep).unsqueeze(1)
        cond_emb = self.cond_obs_emb(cond)
        cond_tokens = torch.cat([time_emb, cond_emb], dim=1)
        memory = self.encoder(self.drop(cond_tokens + self.cond_pos_emb[:, : cond_tokens.shape[1]]))

        token_embeddings = self.input_emb(sample)
        tgt = self.drop(token_embeddings + self.pos_emb[:, : token_embeddings.shape[1]])
        tgt_mask = self.mask[: tgt.shape[1], : tgt.shape[1]] if self.mask is not None else None
        memory_mask = self.memory_mask[: tgt.shape[1], : memory.shape[1]] if self.memory_mask is not None else None
        if self.faithful_attnres is not None:
            x = self._forward_decoder_faithful_attnres(tgt, memory, tgt_mask, memory_mask)
        elif self.sublayer_attnres is not None:
            x = self._forward_decoder_sublayer_attnres(tgt, memory, tgt_mask, memory_mask)
        elif self.attnres is None:
            x = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        elif self.attnres_placement == "branch":
            x = self._forward_decoder_branch_attnres(tgt, memory, tgt_mask, memory_mask)
        else:
            x = self._forward_decoder_block_attnres(tgt, memory, tgt_mask, memory_mask)
        return self.head(self.ln_f(x))

    def _apply_attnres(self, unit_idx: int, history: list[Tensor], candidate: Tensor) -> Tensor:
        assert self.attnres is not None
        x = self.attnres(unit_idx, history + [candidate])
        history.append(x)
        return x

    def _forward_decoder_faithful_attnres(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None,
        memory_mask: Tensor | None,
    ) -> Tensor:
        """Faithful Kimi AttnRes forward (layer-level aggregation points).

        Key fix vs old block/branch: `sources` holds RAW per-unit outputs v_i
        (v_0 = decoder input embedding, v_i = raw layer output). The routed
        result `x` feeds the NEXT layer but is NEVER written back into the
        source history. Each aggregation point uses its own pseudo-query.
        """
        assert self.faithful_attnres is not None
        block_size = self.attnres_block_size
        self.faithful_attnres.begin_forward()

        # v_0 is the decoder input embedding (Kimi includes token embedding).
        raw_outputs: list[Tensor] = [tgt]
        x = tgt

        for layer_idx, layer in enumerate(self.decoder.layers):
            if self.faithful_attnres.mode == "full":
                # Attend over all raw previous outputs before this layer.
                sources = raw_outputs
            else:  # block: completed-block reps + current partial-block sum
                sources = self._block_sources(raw_outputs, block_size)
            x = self.faithful_attnres(layer_idx, sources)
            candidate = layer(tgt=x, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            raw_outputs.append(candidate)  # store RAW output, not routed x

        # Final hidden = aggregate all raw outputs once more (output of network).
        if self.faithful_attnres.mode == "full":
            x = raw_outputs[-1]
        else:
            x = raw_outputs[-1]
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    @staticmethod
    def _block_sources(raw_outputs: list[Tensor], block_size: int) -> list[Tensor]:
        """Build block-level sources: completed block sums + current partial sum.

        raw_outputs = [v_0(embed), v_1, v_2, ...]; v_0 is its own first block rep.
        Within a block, sources are accumulated by standard residual sum.
        """
        if len(raw_outputs) == 1:
            return [raw_outputs[0]]
        embed = raw_outputs[0]
        layer_outs = raw_outputs[1:]  # v_1..v_k
        blocks: list[Tensor] = [embed]
        partial: Tensor | None = None
        for i, v in enumerate(layer_outs):
            partial = v if partial is None else partial + v
            # close block at boundary (every block_size layers)
            if (i + 1) % block_size == 0 and (i + 1) < len(layer_outs):
                blocks.append(partial)
                partial = None
        sources = list(blocks)
        if partial is not None:
            sources.append(partial)
        return sources

    def _forward_decoder_block_attnres(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None,
        memory_mask: Tensor | None,
    ) -> Tensor:
        assert self.attnres is not None
        x = tgt
        history = [x]
        self.attnres.begin_forward()
        for layer_idx, layer in enumerate(self.decoder.layers):
            candidate = layer(tgt=x, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            x = self._apply_attnres(layer_idx, history, candidate)
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    def _forward_decoder_branch_attnres(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None,
        memory_mask: Tensor | None,
    ) -> Tensor:
        assert self.attnres is not None
        x = tgt
        history = [x]
        unit_idx = 0
        self.attnres.begin_forward()
        for layer in self.decoder.layers:
            if layer.norm_first:
                candidate = x + layer._sa_block(layer.norm1(x), tgt_mask, None)
                x = self._apply_attnres(unit_idx, history, candidate)
                unit_idx += 1

                candidate = x + layer._mha_block(layer.norm2(x), memory, memory_mask, None)
                x = self._apply_attnres(unit_idx, history, candidate)
                unit_idx += 1

                candidate = x + layer._ff_block(layer.norm3(x))
                x = self._apply_attnres(unit_idx, history, candidate)
                unit_idx += 1
            else:
                candidate = layer.norm1(x + layer._sa_block(x, tgt_mask, None))
                x = self._apply_attnres(unit_idx, history, candidate)
                unit_idx += 1

                candidate = layer.norm2(x + layer._mha_block(x, memory, memory_mask, None))
                x = self._apply_attnres(unit_idx, history, candidate)
                unit_idx += 1

                candidate = layer.norm3(x + layer._ff_block(x))
                x = self._apply_attnres(unit_idx, history, candidate)
                unit_idx += 1
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    def _forward_decoder_sublayer_attnres(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None,
        memory_mask: Tensor | None,
    ) -> Tensor:
        """Kimi-style sublayer AttnRes forward pass.

        Routes before each sublayer (self-attention, cross-attention, FFN).
        This implements the fine-grained routing from the Kimi paper.
        """
        assert self.sublayer_attnres is not None
        x = tgt
        block_outputs: list[Tensor] = []
        partial_block: Tensor | None = None
        block_size = self.sublayer_attnres.block_size

        for layer_idx, layer in enumerate(self.decoder.layers):
            is_block_start = (layer_idx % block_size == 0)
            if is_block_start:
                if partial_block is not None:
                    block_outputs.append(partial_block)
                partial_block = None

            if layer.norm_first:
                # Route before self-attention
                if self.attnres_route_before_self_attn and (block_outputs or partial_block is not None):
                    x = self.sublayer_attnres(block_outputs, partial_block)

                sa_out = layer._sa_block(layer.norm1(x), tgt_mask, None)
                if partial_block is None:
                    partial_block = sa_out
                else:
                    partial_block = partial_block + sa_out

                # Route before cross-attention (optional)
                if self.attnres_route_before_cross_attn:
                    x = self.sublayer_attnres(block_outputs, partial_block)
                else:
                    x = partial_block

                ca_out = layer._mha_block(layer.norm2(x), memory, memory_mask, None)
                partial_block = partial_block + ca_out

                # Route before FFN
                if self.attnres_route_before_ffn:
                    x = self.sublayer_attnres(block_outputs, partial_block)
                else:
                    x = partial_block

                ff_out = layer._ff_block(layer.norm3(x))
                partial_block = partial_block + ff_out
            else:
                # Route before self-attention
                if self.attnres_route_before_self_attn and (block_outputs or partial_block is not None):
                    x = self.sublayer_attnres(block_outputs, partial_block)

                sa_out = layer._sa_block(x, tgt_mask, None)
                sa_out = layer.norm1(sa_out if partial_block is None else partial_block + sa_out)
                if partial_block is None:
                    partial_block = sa_out
                else:
                    partial_block = sa_out

                # Route before cross-attention (optional)
                if self.attnres_route_before_cross_attn:
                    x = self.sublayer_attnres(block_outputs, partial_block)
                else:
                    x = partial_block

                ca_out = layer._mha_block(x, memory, memory_mask, None)
                ca_out = layer.norm2(partial_block + ca_out)
                partial_block = ca_out

                # Route before FFN
                if self.attnres_route_before_ffn:
                    x = self.sublayer_attnres(block_outputs, partial_block)
                else:
                    x = partial_block

                ff_out = layer._ff_block(x)
                ff_out = layer.norm3(partial_block + ff_out)
                partial_block = ff_out

            x = partial_block

        # Append last block
        if partial_block is not None:
            block_outputs.append(partial_block)

        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x
