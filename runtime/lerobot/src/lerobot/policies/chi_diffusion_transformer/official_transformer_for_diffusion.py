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

"""Official real-stanford Diffusion Policy Transformer trunk.

This is a local adapter of real-stanford/diffusion_policy's
`diffusion_policy/model/diffusion/transformer_for_diffusion.py`, with import
paths adapted so the exact trunk can be used inside the LeRobot training and
evaluation pipeline.
"""

import logging
import math

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class OfficialTransformerForDiffusion(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int | None = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        t_main = horizon
        t_cond = 1
        if not time_as_cond:
            t_main += 1
            t_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            t_cond += n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, t_main, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if t_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, t_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
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
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)
        else:
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

        if causal_attn:
            mask = (torch.triu(torch.ones(t_main, t_main)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                t_idx, s_idx = torch.meshgrid(torch.arange(t_main), torch.arange(t_cond), indexing="ij")
                mask = t_idx >= (s_idx - 1)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.T = t_main
        self.T_cond = t_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            for name in ["in_proj_bias", "bias_k", "bias_v"]:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, OfficialTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, _param in module.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith("bias") or param_name.startswith("bias"):
                    no_decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_param_name)

        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"
        )

        return [
            {
                "params": [param_dict[param_name] for param_name in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        cond: torch.Tensor | None = None,
        **kwargs,
    ):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        input_emb = self.input_emb(sample)

        if self.encoder_only:
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            num_tokens = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :num_tokens, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            num_cond_tokens = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :num_cond_tokens, :]
            x = self.drop(cond_embeddings + position_embeddings)
            memory = self.encoder(x)

            token_embeddings = input_emb
            num_tokens = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :num_tokens, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        return self.head(x)
