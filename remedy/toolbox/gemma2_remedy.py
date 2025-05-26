# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The vLLM team.
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

""" ReMedy vllm arch file with Gemma2 Models. 
    Github Repo: https://github.com/Smu-Tan/Remedy
    Please cite our paper when using it:
        @article{tan2025remedy,
        title={Remedy: Learning Machine Translation Evaluation from Human Preferences with Reward Modeling},
        author={Tan, Shaomu and Monz, Christof},
        journal={arXiv preprint arXiv:2504.13630},
        year={2025}
}
"""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.gemma2 import Gemma2Model
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix


class Gemma2ForSequenceClassification(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Gemma2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # hidden_states from Qwen2Model has been reduced,
        # the input of score layer is not parallelized.
        self.score = RowParallelLinear(config.hidden_size,
                                       config.num_labels,
                                       quant_config=quant_config,
                                       input_is_parallel=False,
                                       bias=False,
                                       prefix=maybe_prefix(prefix, "score"))
        
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=False,
            softmax=False) # original was softmax=True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        
        logits, _ = self.score(hidden_states)

        return logits
    

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["lm_head."])
        loader.load_weights(weights)
