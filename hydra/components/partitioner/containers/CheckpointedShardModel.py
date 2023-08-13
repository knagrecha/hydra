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
# ==============================================================================

import torch.nn as nn
from hydra.utilities import get_free_space

import gc

import torch

"""
    This is the PyTorch module that wraps model layers to form a model shard.
    It runs a simple pass through the layers in sequence.

"""


class CheckpointedShardModel(nn.Module):
    def __init__(self, layers):
        self.layers = layers
        super(CheckpointedShardModel, self).__init__()
        
        for idx, layer in enumerate(layers):
            self.add_module("Module_{}".format(idx), layer)

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint_sequential(self.layers, len(self.layers), x)
