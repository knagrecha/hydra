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

from .TensorPartitioner import tensor_partitioner

import hydra.components.executor as executor

import math

class Pilot():
    def __init__(self):
        self.selected_device = self.select_device()
        self.type="pilot"
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

        total_time = 0
        
        all_layers = list(model.children())
        
        self.selected_device, self.selected_device_index = self.select_device()


        if (double_buffer != 0):
            buffer_space = torch.zeros((double_buffer, double_buffer)).to(self.selected_device)
        
        # According to the mandatory dataloader specification, the last return item is the label
        true_labels = test_batch[-1]
        
        # The rest of the return is the batch
        shard_batch_input = test_batch[0:len(test_batch)-1]
        if (len(shard_batch_input) == 1):
            shard_batch_input = shard_batch_input[0]

        
        # Place the batch on-device
        batch_input = move_batch_to_device(shard_batch_input, self.selected_device) 
        
        
        partitioned_layers = []
        partitioning_index = 0
        
        shard_count = 0
        
        intermediate_activations = [] 
        partition_indices = [] 
        
      
        
        if verbose == 1:
            print("======================Partitioning for {} layers======================".format(len(all_layers)))
            
        while partitioning_index < (len(all_layers)):
            
            batch_input = move_batch_to_device(batch_input, self.selected_device)      
            #print(get_free_space(self.selected_device_index))
            oom = False
          
            try:
                
                all_layers[partitioning_index] = all_layers[partitioning_index].to(self.selected_device)
                partitioned_layers.append(all_layers[partitioning_index])

                if (isinstance(batch_input, tuple) or isinstance(batch_input, list)):
                    out = partitioned_layers[-1](*batch_input)
                else:
                    out = partitioned_layers[-1](batch_input)
                                 
                grads = []
                new_out = []
                if not (isinstance(out, torch.Tensor)):

                    # Create a list of differentiable outputs and sample gradients
                    for output in out:
                        if output.requires_grad:
                            grads.append(torch.ones_like(output).to(self.selected_device))
                            new_out.append(output)

                else:
                    # Run a backward pass on the output
                    if (out.requires_grad):
                        new_out.append(out)
                        grads.append(torch.ones_like(out).to(self.selected_device))

                # Run a backward pass, do not discard memory - future partitioning passes will run through the 
                # same tree.
                if (len(new_out)!= 0):
                    torch.autograd.backward(new_out, grads, retain_graph = True)
                    partitioned_layers[-1].zero_grad()

                del new_out
                del grads # Gradients are discarded immediately after use
                intermediate_activations.append(out) # Add the output of this layer as an intermediate activation
                
                if verbose == 1:
                    print("| Splits: {} | Layer Index: {} | Memory {} |".format(partition_indices, partitioning_index, get_free_space(self.selected_device_index)))
                    
                    
                
                # GPU Memory Consumption is complete
                # if we reach this point we can safely assume the pass is safe. 
                # We can now update the input to prepare for the next layer.

                del batch_input # Consider deleting
                if not (isinstance(out, torch.Tensor)):
                    batch_input = [x.cpu().detach().clone() for x in out]
                else:
                    batch_input = out.cpu().detach().clone()
                
                partitioning_index += 1 # Move onto the next layer

            except Exception as e:
                if ("memory" not in str(e)):
                    raise e
                    
                # Caught a memory error! Try partitioning.
                else:
                    oom = True

            if (oom):

                successful_run = False
                roll_back_count = 0
                while not successful_run:
                    try:
                        roll_back_count += 1
                        successful_run = True
                        batch_input = None
                        out = None
                        pioneer_layer = partitioned_layers.pop() # Remove the last added layer
                        pioneer_layer = pioneer_layer.cpu() # Move it back to CPU memory
                        
                        # We do an end-to-end test on the shard. Discard false intermediate activations.
                        del intermediate_activations
                        intermediate_activations = []
                        
                        if (len(partitioned_layers) == 0):
                            partition_count, chosen_dim, all_models, out, time_taken_f, time_taken_b, partitioner_time, merger_time, cut_points = tensor_partitioner(pioneer_layer, 
                                                                            shard_batch_input, self.selected_device)
                            
                            
                            """
                                Partition_Count = how many divisions to make?
                                chosen_dim = what dimension do we cut across? 
                                all_models = each cloned copy of the conv kernel for each partition
                                out = the merged outputs of each copy
                                time_taken_f = list of each forward shard's time
                                time_taken_b = list of each backward shard's time
                                partitioner_time = time it takes to partition
                                merger_time = timer it takes to merge outputs
                                cut_points = where the gradients will be cut and need to be split going backwards
                            """
                            
                            # Create the Forward Splitter
                            forward_partition_executor = executor.ForwardConvSplitter(shard_count, 
                                            partition_count, chosen_dim, layer.kernel_size, layer.padding, layer.stride)
                            
                            forward_partitioner_task = ShardedTask(None, forward_partition_executor, "f",
                                                                                partitioner_time, shard_count, lr, {-1})
                            
                            
                            # Create the forward Merger
                            forward_merger_executor = executor.ForwardConvMerger(shard_count, chosen_dim)
                            merger_requests = set([x for x in range(-partition_count, 0)])
                            # requests from the earliest generated (i.e. partition_output_0 to the newest (-1)).
                            
                            forward_merger_task = ShardedTask(None, forward_merger_executor, "f",
                                                                                merger_time, shard_count, lr, merger_requests)
                            # Create the Forward and Backward going shards
                            forward_partition_shards = []
                            backward_partition_shards = []
                            
                            for model_idx in range(partition_count):
                                forward_partition_shards.append(ShardedTask(all_models[model_idx], 
                                                                    executor.Forward(shard_count), 
                                                                    "f", time_taken_f, lr, {-partition_count})
                                # Creates a shard requesting input of - partition_count. Keep in mind that the forward
                                # going queue will be appended to by each shard, so the offset is consistent each time.
                                                                
                                                                
                                backward_partition_shards.append(ShardedTask(all_models[model_idx], 
                                                                    executor.Backward(shard_count), 
                                                                    "b", time_taken_b, lr, {model_idx - partition_count})
                                # Creates a shard requesting inputs ranging from -partition_count to -1.
                            
                                                            
                            # Create the Backward Splitter
                            backward_partition_executor = executor.BackwardGradientSplitter(shard_count, 
                                            partition_count, chosen_dim, cut_points)
                            
                            backward_partitioner_task = ShardedTask(None, backward_partition_executor, "b",
                                                                                partitioner_time, shard_count, lr, {-1})
                            
                            
                            # Create the Backward Merger
                            backward_merger_executor = executor.BackwardGradientMerger(shard_count, chosen_dim)
                            merger_requests = set([-x for x in range(1, partition_count+1)])
                            
                            backward_merger_task = ShardedTask(None, backward_merger_executor, "b",
                                                                                merger_time, shard_count, lr, merger_requests)
                             
                                                                 
   
                        else:
                            oom = False

                            shard_batch_input = move_batch_to_device(shard_batch_input, self.selected_device)

                            start_f = timer() # used for scheduler
                            model = ShardModel(nn.ModuleList(partitioned_layers)) # Create a shard-model
                            model.to(self.selected_device)
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

                            batch_input = shard_batch_input
                            
                            # Create new shards
                            if partitioning_index == (len(all_layers)) - 1:
                                forward_shards.append(ShardedTask(model, executor.ForwardLoss(shard_count), 
                                                                        "f", end_f-start_f, shard_count, lr, {-1}))
                            
                            else:
                                forward_shards.append(ShardedTask(model, executor.Forward(shard_count), 
                                                                            "f", end_f-start_f, shard_count, lr, {-1}))
                                backward_shards.append(ShardedTask(model, executor.Backward(shard_count), 
                                                                            "b", end_b-start_b, shard_count, lr, {-1}))
                        
                    except Exception as e:
                        if ("memory" not in str(e)):
                            raise e
                        # Caught a memory error! Try rolling back another layer.
                        else:
                            if verbose == 1:
                                print("Partition failed...rolling back another layer")
                                shard_batch_input = move_batch_to_device(shard_batch_input, torch.device("cpu"))
                                model = None
                                gradients = None
                                out = None
                            successful_run = False

                
                shard_count+=1

                total_time = total_time + (end_f - start_f) + (end_b - start_b)
                partitioned_layers = []
                
                partitioning_index -= roll_back_count # Return to partitioning from the appropriate location
                partition_indices.append(partitioning_index) # Record partition location

                
        # While loop has terminated, but we have not sharded the last set of layers yet
        if (len(partitioned_layers) != 0):
            
            del intermediate_activations # Consider deleting
            intermediate_activation = []
            
        
            shard_batch_input = move_batch_to_device(shard_batch_input, self.selected_device)
            start_f = timer() # used for scheduler

            model = ShardModel(nn.ModuleList(partitioned_layers)) # Create shard model
            model.to(self.selected_device)

            out = model(shard_batch_input) # Run a forward pass
            end_f = timer()

            start_b = timer() # used for scheduler
            
            true_labels = torch.ones_like(out)
            torch.autograd.backward(out, true_labels)
            model.zero_grad()
            
            #del loss
            del true_labels
            del out
            

            del batch_input

            model.cpu()  # this is an inplace operation
            end_b = timer()

            forward_shards.append(ShardedTask(model, executor.ForwardLoss(shard_count), "f", end_f - start_f, shard_count, lr, {-1}) )
          
            total_time = total_time + (end_f - start_f) + (end_b - start_b)

            
        if verbose == 1:
            print("==============Number of Shards: {}======================".format(len(forward_shards)))
            print("=======Anticipated Minibatch Times: {:.2f}s=======".format(total_time))
            
        backward_shards.reverse()
        
        if (double_buffer != 0):
            del buffer_space
            torch.cuda.empty_cache()
        if (forward_shards[-1].executor.type != "Forward Loss"):
            forward_shards[-1].executor = ForwardLoss(shard_count)
            
        return forward_shards, backward_shards, total_time

                #print([get_free_space(x) for x in available_devices])

