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
import math

"""
    Tensor parallel splitter module.
"""

class ForwardConvSplitter():
    def __init__(self, idx, split_count, chosen_dim, kernel_size, padding, stride):
        self.type="Splitter"
        self.idx = idx
        self.split_count = split_count
        
        self.kernel_extension_prior = math.floor((kernel_size[chosen_dim-2]-1) / 2)
        self.kernel_extension_latter = math.ceil((kernel_size[chosen_dim-2]-1) / 2)
        self.chosen_dim = chosen_dim

    def run(self, model, batch_input, device):
        with torch.no_grad():

            # Splitter recives single batch input
            batch_input = batch_input[0]

            remainder = batch_input.shape[self.chosen_dim] % self.split_count
            previous_partition = 0
            partition_dimensional_value = 0

            partitions = []

            for partition_index in range(self.split_count):

                # This code produces roughly even partition points

                if (partition_index) < remainder:
                    partition_dimensional_value += math.floor(batch.shape[chosen_dim] / partition_count) + 1
                else:
                    partition_dimensional_value += math.floor(batch.shape[chosen_dim] / partition_count)


                partition = None

                # left-most/top-most/forward-most side of image
                if partition_index == 0:
                    print("SLICING FROM {} to {}".format(previous_partition, partition_dimensional_value+kernel_extension_latter))
                    slice_array = [slice(None) for x in range(len(batch.shape))]
                    slice_array[chosen_dim] = slice(None, partition_dimensional_value+kernel_extension_latter)
                    slice_array = tuple(slice_array)
                    partition = batch[slice_array]
                    previous_partition = partition_dimensional_value

                # right-most/bottom-most/back-most side of image
                elif partition_index == partition_count - 1:
                    print("SLICING FROM {} to {}".format(previous_partition-kernel_extension_prior, partition_dimensional_value))
                    slice_array = [slice(None) for x in range(len(batch.shape))]
                    slice_array[chosen_dim] = slice(previous_partition-kernel_extension_prior, None)
                    slice_array = tuple(slice_array)
                    partition = batch[slice_array]

                    previous_partition = partition_dimensional_value

                # innards of image
                else:
                    print("SLICING FROM {} to {}".format(previous_partition-kernel_extension_prior, partition_dimensional_value+kernel_extension_latter))


                    slice_array = [slice(None) for x in range(len(batch.shape))]
                    slice_array[chosen_dim] = slice(previous_partition-kernel_extension_prior, partition_dimensional_value+kernel_extension_latter)
                    slice_array = tuple(slice_array)

                    partition = batch[slice_array]
                    previous_partition = partition_dimensional_value

                partitions.append(partition) # partitions are being generated!


            return partitions
                    
                 