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
from hydra.utilities import get_free_space, move_batch_to_device
import numpy as np
from timeit import default_timer as timer

import torch
from hydra.components.partitioner.containers import ShardModel, ShardedTask
import traceback

from hydra.components.executor import Forward, ForwardLoss, Backward

import math

class Presharded():
    def __init__(self, partitions):
        self.selected_device, self.selected_device_index = self.select_device()
        self.type="presharded"
        self.partitions = partitions
    """
        Simply selects the device to run partitioning tests with.
        For heterogeneous settings, use the device with the smallest memory as the
        partitioning tester.
    
    """
    def select_device(self):
        gpu_count = torch.cuda.device_count()
        available_devices = list(range(gpu_count))
        free_spaces = [get_free_space(x) for x in available_devices]
        
        device_idx = np.argmin(free_spaces)
        return torch.device("cuda:"+str(device_idx)), device_idx
    

    
    """
        The partitioning function. Takes as input a model along with various
        critical runtime inputs (loss function, sample batch).
        
        It also uses the specified double-buffering space to protect a portion
        of memory during partitioning.
        
        TODO: Change all_layers to use model.modules() instead of model.children()
    
    """

    def shard(self, model, criterion, test_batch, double_buffer, lr, verbose):

        forward_shards = []
        backward_shards = []
        true_labels = test_batch[-1]
        
        # The rest of the return is the batch
        shard_batch_input = test_batch[0:len(test_batch)-1]
        batch_input = shard_batch_input 
        # Place the batch on-device
        total_time = 0
        
        all_layers = list(model.children())
        
        partitioning_index = 0
        for idx, partition in enumerate(self.partitions):
            partitioned_layers = []
            while partitioning_index < partition:                    
                partitioned_layers.append(all_layers[partitioning_index])
                partitioning_index += 1
            start_f = timer() # used for schduler
            model = ShardModel(nn.ModuleList(partitioned_layers)) # Create a shard-model
            model.to(self.selected_device)
            shard_batch_input = move_batch_to_device(shard_batch_input, self.selected_device)
            out = model(shard_batch_input)
            end_f = timer()
            start_b = timer() # used for scheduler
            gradients = out.detach().clone()
            torch.autograd.backward(out, gradients)
            model.zero_grad()
                        
            del gradients
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            model.cpu()  # this is an inplace operation

            end_b = timer()
                        
                        
            if not (isinstance(out, torch.Tensor)):
                shard_batch_input = [x.cpu().detach().clone() for x in out]
            else:
                shard_batch_input = out.cpu().detach().clone()
            del out
                        
            params = sum(p.numel() for p in model.parameters())
            print("NEW SHARD - {} PARAMETERS".format(params))
            forward_shards.append(ShardedTask(model, Forward(idx), "f", end_f-start_f, idx, lr))
            backward_shards.append(ShardedTask(model, Backward(idx), "b", end_b-start_b, idx, lr))

            total_time = total_time + (end_f - start_f) + (end_b - start_b)
            partitioned_layers = []
            torch.cuda.empty_cache() # not necessary, just makes memory debugging easier to view
            if verbose == 1:
                print("Free Memory: {}".format(get_free_space(self.selected_device_index)))

        if verbose == 1:
            print("==============Number of Shards: {}======================".format(len(forward_shards)))
            print("=======Anticipated Minibatch Times: {:.2f}s=======".format(total_time))
            
        backward_shards.reverse()
        backward_shards.pop(0) # The forward and backward shards share their last and first shards respectively, only need 1

        if (forward_shards[-1].executor.type != "Forward Loss"):
            forward_shards[-1].executor = ForwardLoss(len(self.partitions))
            
        return forward_shards, backward_shards, total_time
