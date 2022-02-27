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
    Tensor parallel backward splitter module. Needs to ensure that the reverse splits are identical to the forward splits, i.e.
    each partition is the same shape as the corresponding output
"""

class BackwardGradientSplitter():
    def __init__(self, idx, split_count, chosen_dim, cut_points):
        self.type="Splitter"
        self.idx = idx
        self.split_count = split_count
        self.chosen_dim = chosen_dim
        self.cut_points = cut_points

    def run(self, model, batch_input, device):
        with torch.no_grad():

            # Splitter recives single batch input
            batch_input = batch_input[0]

            for idx, cut_point in enumerate(self.cut_points):
                partition = None
                slice_array = [slice(None) for x in range(len(batch.shape))]
                if (idx == 0):
                    slice_array[chosen_dim] = slice(None, cut_point)
                elif (idx == len(self.cut_points) - 1):
                    slice_array[chosen_dim] = slice(cut_point, None)
                else:
                    slice_array[chosen_dim] = slice(self.cut_points[idx-1], cut_point)
                slice_array = tuple(slice_array)
                partition = batch[slice_array]
                partitions.append(partition) # partitions are being generated!


            return partitions
                    
                 