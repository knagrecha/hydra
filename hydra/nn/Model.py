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
from .utilities import get_free_space
import numpy as np
import gc
from timeit import default_timer as timer

import torch
from .Container import NNContainer
import traceback



class ShardedTask():

    def __init__(self, model, direction, time_taken, idx, lr):
        self.lr = lr
        self.model = model
        self.direction = direction
        self.time_cost = time_taken
        self.idx = idx
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)



class Model():
    def __init__(self, model):
        self.f_shards = []
        self.b_shards = []
        # this info is nice to have for the scheduler
        self.shard_forward_times = []
        self.shard_backward_times = []
        self.shard_times = []
        self.total_time = 0
        self.verbose = 0

        self.layers = list(model.children()) # ideally we'd use model.modules. But that could lead to
                                             # issues in our sharding algo. TODO

    
    def setup(self, true_criterion, batch_orig, buffer, lr):
        
        available_gpus = torch.cuda.device_count()
        available_devices = list(range(available_gpus))
        free_spaces = [get_free_space(x) for x in available_devices]
        
        device_idx = np.argmin(free_spaces)
        device = torch.device("cuda:"+str(device_idx))

        if (buffer != None):
            buffer_arr = torch.zeros((buffer, buffer)).to(device)
        else:
            buffer_arr = torch.zeros((15000, 15000)).to(device)

        true_labels = batch_orig[-1]
        batch_orig = batch_orig[0:len(batch_orig)-1]
        
        batch = [x.to(device) for x in batch_orig]

        
        list_of_layers = []

        true_layer_index = 0
        shard_idx = 0
        #print("Free memory: " + str(get_free_space()))
        sequence_grads = []
        sequence_outs = []
        split_indices = []
        print("======================Partitioning======================")
        while true_layer_index < (len(self.layers)):
            if not (isinstance(batch, torch.Tensor)):
                batch = [x.to(device) for x in batch_orig]
            else:
                batch = batch.to(device)
            
            
            oom = False
            oom_override = False
            #if (self.verbose == 1):
            #    print("Current Memory Free {0} MB\t".format(get_free_space(device_idx)/(1024*1024)))
            
            
            try:
                list_of_layers.append(self.layers[true_layer_index])
                self.layers[true_layer_index] = self.layers[true_layer_index].to(device)
                
                if (isinstance(batch, tuple) or isinstance(batch, list)):
                    #print (mod)
                    out = self.layers[true_layer_index](*batch)
                else:
                    out = self.layers[true_layer_index](batch)
                
                
                if true_layer_index == len(self.layers) - 1:
                    
                    del sequence_outs

                    sequence_outs = []

           

                    if not (isinstance(batch_orig, torch.Tensor)):
                        batch_orig = [x.to(device) for x in batch_orig]
                    else:
                        batch_orig = batch_orig.to(device)

                    start_f = timer() # used for scheduler

                    model = NNContainer(nn.ModuleList(list_of_layers))
                    model.to(device)

                    out = model(batch_orig)

                    end_f = timer()
                    self.shard_forward_times.append(end_f-start_f)


                    start_b = timer() # used for scheduler
                    if not isinstance(true_labels, torch.Tensor):
                        true_labels = [x.to(device, non_blocking=True) for x in true_labels]
                    else:
                        true_labels = true_labels.to(device, non_blocking=True)


                    loss = true_criterion(out, true_labels)
                    loss.backward()

                    model.zero_grad()
                    del loss
                    del true_criterion
                    del true_labels
                    del out
                    if (isinstance(batch, list) or isinstance(batch, tuple)):
                        batch = [x.cpu() for x in batch]
                    del batch

                    model.to(torch.device("cpu"))  # this is an inplace operation

                    end_b = timer()
                    self.f_shards.append(ShardedTask(model, "f", end_f - start_f, shard_idx, lr))
                    self.b_shards.append(ShardedTask(model, "b", end_b - start_b, shard_idx, lr))

                    self.total_time = self.total_time + (end_f - start_f) + (end_b - start_b)

                    self.b_shards.reverse()
                    self.b_shards.pop(0)
                    
                    if (buffer_arr != None):
                        del buffer_arr
                        torch.cuda.empty_cache()
  
                    true_layer_index+=1
                else:
                
                    if not (isinstance(out, torch.Tensor)):
                        grads = []
                        new_out = []
                        for output in out:
                            if output.requires_grad:
                                grads.append(torch.ones_like(output).to(device))
                                new_out.append(output)
                        if (len(new_out)!= 0):
                            torch.autograd.backward(new_out, grads, retain_graph = True)
                            self.layers[true_layer_index].zero_grad()

                    else:
                        if (out.requires_grad):
                            grads = []
                            grads.append(torch.ones_like(out).to(device))
                            torch.autograd.backward(out, grads, retain_graph = True)
                            self.layers[true_layer_index].zero_grad()



                    sequence_outs.append(out)
                    #sequence_grads.append(grads)
                    print("| Splits: {} | Layer Index: {} | Memory {} |".format(split_indices, true_layer_index, [get_free_space(x) for x in available_devices]), end='\r', flush=True)

                    # GPU Memory Consumption is complete
                    # if we reach this point we can safely assume the pass is safe. (i.e. batch will soon be replaced by out - why hold onto it?)
                    # so if we DO NOT reach this point, batch is unmodified (i.e. ready for a new pass in a new subset)

                    del batch
                    if not (isinstance(out, torch.Tensor)):
                        batch = [x.clone().detach().cpu() for x in out]
                    else:
                        batch = out.clone().detach().cpu()
                    true_layer_index+=1

            except Exception as e:
                #print(e)
                #traceback.print_exc()
                oom = True

            if (oom):
                true_layer_index -=1
                split_indices.append(true_layer_index)
                pioneer_layer = list_of_layers.pop()
                pioneer_layer = pioneer_layer.cpu()
                if (len(list_of_layers) == 0):
                    raise RuntimeError("Your minimum defined module's size is too large! Try chunking your modules into smaller pieces?")


                del sequence_outs
                sequence_outs = []

                oom = False
                if not (isinstance(batch_orig, torch.Tensor)):
                    batch_orig = [x.to(device) for x in batch_orig]
                else:
                    batch_orig = batch_orig.to(device)
                start_f = timer() # used for scheduler

                model = NNContainer(nn.ModuleList(list_of_layers))
                model.to(device)


                out = model(batch_orig)
                end_f = timer()

                start_b = timer() # used for scheduler

                labels = out.detach().clone()
                criterion = nn.MSELoss()
                loss = criterion(out, labels)
                loss.backward()
                model.zero_grad()
                if not (isinstance(out, torch.Tensor)):
                    batch_orig = [x.cpu().detach().clone() for x in out]
                else:
                    batch_orig = out.cpu().detach().clone()

                del out

                del loss
                del criterion
                del labels

                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                model.cpu()  # this is an inplace operation


                end_b = timer()

                #model.share_memory()
                self.f_shards.append(ShardedTask(model, "f", end_f-start_f, shard_idx, lr))
                self.b_shards.append(ShardedTask(model, "b", end_b-start_b, shard_idx, lr))
                shard_idx+=1

                self.total_time = self.total_time + (end_f - start_f) + (end_b - start_b)



                list_of_layers = []
                list_of_layers.append(pioneer_layer)
        print()
        print("=======Anticipated Minibatch Times: {:.2f}s=======".format(self.total_time))

                #print([get_free_space(x) for x in available_devices])

