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

import torch
from torch import Tensor, nn


class AttentionResiduals(nn.Module):
    """Depth-wise aggregation over Chi decoder block outputs.

    This module is inserted between complete decoder blocks. It does not alter
    the self-attention, cross-attention, or feed-forward residual branches inside
    a TransformerDecoderLayer.
    """

    _VALID_MODES = {"full", "block", "uniform", "scalar"}

    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        mode: str = "block",
        block_size: int = 4,
        warm_start: bool = True,
        warm_start_bias: float = 8.0,
        query_init: str = "zero",
        query_init_std: float = 0.02,
        return_weights: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in self._VALID_MODES:
            raise ValueError(f"`mode` must be one of {sorted(self._VALID_MODES)}. Got {mode}.")
        if block_size <= 0:
            raise ValueError(f"`block_size` must be positive. Got {block_size}.")
        if query_init not in {"zero", "normal"}:
            raise ValueError(f"`query_init` must be one of {{'zero', 'normal'}}. Got {query_init}.")
        if query_init_std <= 0:
            raise ValueError(f"`query_init_std` must be positive. Got {query_init_std}.")
        if warm_start_bias < 0:
            raise ValueError(f"`warm_start_bias` must be non-negative. Got {warm_start_bias}.")

        self.dim_model = dim_model
        self.num_layers = num_layers
        self.mode = mode
        self.block_size = block_size
        self.warm_start = warm_start
        self.warm_start_bias = warm_start_bias
        self.query_init = query_init
        self.query_init_std = query_init_std
        self.return_weights = return_weights
        self.eps = eps

        if mode in {"full", "block"}:
            self.queries = nn.Parameter(torch.empty(num_layers, dim_model))
        else:
            self.register_parameter("queries", None)

        if mode == "scalar":
            self.scalar_logits = nn.Parameter(torch.empty(num_layers, num_layers + 1))
        else:
            self.register_parameter("scalar_logits", None)

        self.logit_bias = nn.Parameter(torch.empty(num_layers, num_layers + 1))
        self.last_weights: list[Tensor] = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.queries is not None:
            if self.query_init == "zero":
                nn.init.zeros_(self.queries)
            else:
                nn.init.normal_(self.queries, mean=0.0, std=self.query_init_std)
        if self.scalar_logits is not None:
            nn.init.zeros_(self.scalar_logits)
        nn.init.zeros_(self.logit_bias)
        if self.warm_start:
            with torch.no_grad():
                for layer_idx in range(self.num_layers):
                    self.logit_bias[layer_idx, layer_idx + 1] = self.warm_start_bias
        self.last_weights = []

    def begin_forward(self) -> None:
        if self.return_weights:
            self.last_weights = []

    def forward(self, layer_idx: int, history: list[Tensor]) -> Tensor:
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"`layer_idx` must be in [0, {self.num_layers}). Got {layer_idx}.")
        if len(history) < 2:
            raise ValueError("`history` must contain the decoder input and at least one layer output.")

        selected = self._selected_indices(len(history))
        states = torch.stack([history[index] for index in selected], dim=0)
        scores = self._scores(layer_idx, selected, states)
        weights = scores.softmax(dim=0)
        output = (weights.unsqueeze(-1) * states).sum(dim=0)

        if self.return_weights:
            self.last_weights.append(weights.detach().mean(dim=(1, 2)).cpu())

        return output

    def _selected_indices(self, history_len: int) -> list[int]:
        if self.mode != "block":
            return list(range(history_len))

        current_index = history_len - 1
        selected = [0]
        for index in range(1, current_index):
            if (index - 1) % self.block_size == self.block_size - 1:
                selected.append(index)
        selected.append(current_index)
        return selected

    def _scores(self, layer_idx: int, selected: list[int], states: Tensor) -> Tensor:
        bias = self.logit_bias[layer_idx, selected].view(-1, 1, 1)
        if self.mode == "uniform":
            return states.new_zeros((len(selected), *states.shape[1:3]))
        if self.mode == "scalar":
            assert self.scalar_logits is not None
            logits = self.scalar_logits[layer_idx, selected].view(-1, 1, 1)
            return logits.expand(-1, *states.shape[1:3]) + bias

        assert self.queries is not None
        keys = self._rms_norm(states)
        query = self.queries[layer_idx].to(dtype=states.dtype)
        dynamic_scores = torch.einsum("hbtd,d->hbt", keys, query) / math.sqrt(self.dim_model)
        return dynamic_scores + bias

    def _rms_norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SublayerAttnRes(nn.Module):
    """Kimi-style sublayer AttnRes for DP-T.

    Routes before each sublayer (self-attention, cross-attention, FFN) instead of
    after complete decoder blocks. This is the fine-grained version from the Kimi paper.

    Key differences from block placement:
    - Routes at sublayer level (attention before + MLP before)
    - Uses RMSNorm on key (Kimi official)
    - Zero-initialized query (Kimi official)
    - Block-internal accumulation + inter-block attention
    """

    def __init__(
        self,
        dim_model: int,
        num_layers: int,
        block_size: int = 4,
        use_rmsnorm: bool = True,
        query_init: str = "zero",
        query_init_std: float = 0.02,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_blocks = (num_layers + block_size - 1) // block_size
        self.use_rmsnorm = use_rmsnorm
        self.eps = eps

        self.query = nn.Parameter(torch.zeros(dim_model))
        self.key_norm = RMSNorm(dim_model, eps) if use_rmsnorm else nn.Identity()

        self._init_weights(query_init, query_init_std)

    def _init_weights(self, query_init: str, query_init_std: float) -> None:
        if query_init == "zero":
            nn.init.zeros_(self.query)
        elif query_init == "normal":
            nn.init.normal_(self.query, mean=0.0, std=query_init_std)
        else:
            raise ValueError(f"`query_init` must be one of {{'zero', 'normal'}}. Got {query_init}.")

    def forward(
        self,
        block_outputs: list[Tensor],
        partial_block: Tensor | None,
    ) -> Tensor:
        if not block_outputs and partial_block is None:
            raise ValueError("At least one of block_outputs or partial_block must be provided.")

        states = []
        if block_outputs:
            states.extend(block_outputs)
        if partial_block is not None:
            states.append(partial_block)

        V = torch.stack(states, dim=0)
        K = self.key_norm(V)
        query = self.query.to(dtype=V.dtype)
        logits = torch.einsum("d, n b t d -> n b t", query, K) / math.sqrt(self.dim_model)
        weights = logits.softmax(dim=0)
        output = (weights.unsqueeze(-1) * V).sum(dim=0)

        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Kimi official)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class FaithfulAttnRes(nn.Module):
    """Faithful Kimi-style Attention Residuals for the Chi DP-T decoder.

    This is a from-scratch reimplementation that fixes the mechanism-level bugs
    found in `AttentionResiduals` (block/branch) and `SublayerAttnRes`:

    Kimi AttnRes core constraints (from arXiv:2603.15031 README/paper):
      1. Each aggregation point l owns an INDEPENDENT learned pseudo-query w_l.
         (old block/branch shared a per-unit query but routed mixed states;
          old sublayer used a SINGLE global query -> no layer specialization.)
      2. The aggregation sources v_i are the RAW per-unit outputs
         (v_0 = decoder input embedding, v_i = f_i(.)), NOT the routed/mixed
         hidden states. The routed output must NOT be written back into history.
      3. Pseudo-queries are ZERO-initialized so initial attention weights are
         uniform across sources (reduces to equal-weight average at init).
         No warm-start logit bias.
      4. RMSNorm is applied on the KEY side to stop large-magnitude sources from
         dominating the softmax.

    Modes:
      - "full":  every unit attends over all previous raw outputs (O(L) sources).
      - "block": layers grouped into blocks; within a block standard residual
                 accumulation, across blocks softmax attention over completed
                 block reps + current partial-block sum (O(N) sources).

    Aggregation unit granularity is controlled by the decoder forward loop; this
    module is unit-agnostic and only needs `num_units` (number of aggregation
    points) at construction.
    """

    _VALID_MODES = {"full", "block"}

    def __init__(
        self,
        dim_model: int,
        num_units: int,
        mode: str = "full",
        use_rmsnorm: bool = True,
        query_init: str = "zero",
        query_init_std: float = 0.02,
        return_weights: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in self._VALID_MODES:
            raise ValueError(f"`mode` must be one of {sorted(self._VALID_MODES)}. Got {mode}.")
        if query_init not in {"zero", "normal"}:
            raise ValueError(f"`query_init` must be one of {{'zero', 'normal'}}. Got {query_init}.")
        if query_init_std <= 0:
            raise ValueError(f"`query_init_std` must be positive. Got {query_init_std}.")

        self.dim_model = dim_model
        self.num_units = num_units
        self.mode = mode
        self.use_rmsnorm = use_rmsnorm
        self.return_weights = return_weights
        self.eps = eps

        # One INDEPENDENT pseudo-query per aggregation point (Kimi constraint #1).
        self.queries = nn.Parameter(torch.empty(num_units, dim_model))
        # RMSNorm on the key side (constraint #4). Shared norm across sources is
        # fine: it is a per-vector normalization, not a per-source parameter.
        self.key_norm = RMSNorm(dim_model, eps) if use_rmsnorm else nn.Identity()

        self.last_weights: list[Tensor] = []
        self.reset_parameters(query_init, query_init_std)

    def reset_parameters(self, query_init: str = "zero", query_init_std: float = 0.02) -> None:
        if query_init == "zero":
            nn.init.zeros_(self.queries)  # uniform initial weights (constraint #3)
        else:
            nn.init.normal_(self.queries, mean=0.0, std=query_init_std)
        self.last_weights = []

    def begin_forward(self) -> None:
        if self.return_weights:
            self.last_weights = []

    def forward(self, unit_idx: int, sources: list[Tensor]) -> Tensor:
        """Aggregate `sources` (raw v_i) with the pseudo-query of `unit_idx`.

        Args:
            unit_idx: index of the current aggregation point (selects w_l).
            sources:  list of raw representations [B, T, D]; for "full" mode this
                      is [v_0, v_1, ..., v_{l-1}]; for "block" mode it is
                      [b_0, ..., b_{n-1}, partial_block].
        Returns:
            Aggregated hidden state [B, T, D].
        """
        if not 0 <= unit_idx < self.num_units:
            raise ValueError(f"`unit_idx` must be in [0, {self.num_units}). Got {unit_idx}.")
        if len(sources) < 1:
            raise ValueError("`sources` must contain at least one representation.")

        v = torch.stack(sources, dim=0)  # [S, B, T, D]
        k = self.key_norm(v)
        query = self.queries[unit_idx].to(dtype=v.dtype)
        logits = torch.einsum("d, s b t d -> s b t", query, k) / math.sqrt(self.dim_model)
        weights = logits.softmax(dim=0)  # [S, B, T]
        output = (weights.unsqueeze(-1) * v).sum(dim=0)

        if self.return_weights:
            self.last_weights.append(weights.detach().mean(dim=(1, 2)).cpu())

        return output
