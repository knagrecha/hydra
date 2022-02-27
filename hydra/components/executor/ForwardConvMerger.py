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

from hydra.utilities import delete_batch, move_batch_to_device
import torch

"""
    Tensor parallel merger module.
"""

class ForwardConvMerger():
    def __init__(self, idx, chosen_dim):
        self.type="Forward"
        self.idx = idx
        self.chosen_dim = chosen_dim

    def run(self, model, batch_input, device):
        with torch.no_grad():
            return torch.cat(batch_input, dim=self.chosen_dim)
                    
                 