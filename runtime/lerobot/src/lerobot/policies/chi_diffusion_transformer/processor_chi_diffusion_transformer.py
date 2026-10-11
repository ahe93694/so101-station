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

from typing import Any

import torch

from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_chi_diffusion_transformer import ChiDiffusionTransformerConfig


def make_chi_diffusion_transformer_pre_post_processors(
    config: ChiDiffusionTransformerConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_diffusion_pre_post_processors(config=config, dataset_stats=dataset_stats)
