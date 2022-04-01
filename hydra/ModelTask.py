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


import torch
import gc
from timeit import default_timer as timer
from hydra.utilities import get_free_space
import random
import numpy as np
from torch import multiprocessing as mp
from collections import defaultdict
import copy







def update_shared_parameters(base_model, queue):
    
    print("SETTING UP BASE OPTIMIZERS")
    local_optimizers = {k: torch.optim.SGD(base_model[k].parameters()) for k in base_model.keys()}
    print("DONE SETTING UP BASE OPTIMIZERS")
    while True:
        shard_key = queue.get()
        state_dict = queue.get()
        print("INPUT RECEIVED!")
        for param_x, param_y in zip(base_model[shard_key].parameters(), state_dict):
            param_x._grad = param_y.cpu()
        
        local_optimizers[shard_key].step()





class ModelTask():
    #def __init__(self, name, model, criterion, dataloader, lr, epochs, global_timer=None, use_scaler=False, partitioner=Pilot()):
    
    """
        name: simple identifier used during debugging
        layer_dictionary: identifies layers with numbers
        io_dictionary: maps layers to what they pass data to forward
            Batch components can be referenced as batch_{idx}
        criterions must be part of the layer_dictionary and io_dictionary passed in.
            Labels can be referenced as label_{idx}
        dataloader: PyTorch dataloader for data access
        lr: learning rate
        epochs: total epochs to be run   
    """
    
    
    def __init__(self, name, layer_dictionary, io_dictionary, dataloader, lr, epochs, verbose=0):
        
        
        self.data_parallel_degree = 2
        
        
        """
            Understanding arbitrary model execution requires a "mapping" of inputs and outputs in the model.
            For this purpose, we use three Python dictionaries.
            
            
            layer_dictionary:
                Maps layer indices to actual Torch layer objects.
            
            layer_to_input: 
                The keys are layer indices from the model, and the values are the 
                layer indices that need to be resolved
                for that key to be executed.
                
            layer_to_output: 
                The keys are layer indices from the model, and the values are the layer indices that
                use this layer as inp
            
        """
        self.layer_dictionary = layer_dictionary
        self.layer_to_input_dict = io_dictionary
        
        self.mp_queue = mp.Queue()
        
        self.layer_to_output_dict = defaultdict(list)
        

        self.tensor_dictionary = {}
        self.grad_dictionary = {}
        
        # endpoint keys (criterions) will essentially not appear in
        # any key's value. They do not output to anyone
        
        # essentially invert the dictionary
        for key, value in self.layer_to_input_dict.items():
            for val in value:
                self.layer_to_output_dict[val].append(key)
                
    
        endpoint_keys = self.layer_to_input_dict.keys() - self.layer_to_output_dict.keys()
        for key in endpoint_keys:
            self.grad_dictionary[key] = None

        """
            The layer_to_inputs can be used to create a coarser shard_to_input dict.
                layer_dictionary:
                    Maps indices to ShardTask objects.
                shard_to_input:
                    The keys are shard indices from the model, and the values are the 
                    LAYER (not shard) indices that need to be resolved
                    for that key to be executed.
                    
                shard_to_output:
                    The keys are shard indices from the model, and the values are the LAYERS (not shard)
                    indices that use this shard as input.
            
        """
        
        self.shard_dictionary = {}
        self.shard_to_input_dict = {}
        self.shard_to_output_dict = defaultdict(list)
        
        """
            A list of layer indices that have all their 
            dependencies resolved. At each shard completion, 
            we update resolved dependencies.
        """
        self.candidate_shards = set()
        self.blocked_shards = set()
        self.completed_shards = set() # how many shards have been run
        
        
        self.endpoint_keys = None
        
        """
        
            DP Clones
        
        """
        
        self.dp_shard_dictionary = []
        self.dp_tensor_dictionary = [{} for x in range(self.data_parallel_degree)]
        self.dp_grad_dictionary = [{} for x in range(self.data_parallel_degree)]
        self.dp_candidate_shards = [set() for x in range(self.data_parallel_degree)]
        self.dp_blocked_shards = [set() for x in range(self.data_parallel_degree)]
        self.dp_completed_shards = [set() for x in range(self.data_parallel_degree)]
        
        
        self.global_timer = 0 # used to identify running times
        self.name = name # used in debugging
          
        self.remaining_runtime = 0

        self.lr = lr
        self.total_epochs = epochs
        self.epochs = epochs
        #self.criterion = criterion
        self.batch_size = dataloader.batch_size
        
        self._old_data = dataloader
        self.dataloader = iter(dataloader)
        self.total_length = len(self._old_data)
        self.minibatches_remaining = len(dataloader)
        self.verbose = verbose

        self.mini_batch_time = 0

        self.last_mini_time = 0
        self.last_runtime = 0
        self.last_loss = 0
        
        self.blocked_devices = []
        
    def setup(self, buffer=None, shard_dictionary=None, shard_to_input_dict=None, shard_to_output_dict=None, mini_batch_time=None):
        
        
        if shard_dictionary is not None:
            self.shard_dictionary = shard_dictionary
            self.shard_to_input_dict = shard_to_input_dict
            self.shard_to_output_dict = shard_to_output_dict
            self.mini_batch_time = mini_batch_time
            
            self.endpoint_keys = self.layer_to_input_dict.keys() - self.layer_to_output_dict.keys()
            
            """
                DP Copies
            """

            for d in range(self.data_parallel_degree):
                print("CREATING DP INSTANCE: {}".format(d))
                new_sh_dict = {k: task.copy() for k, task in shard_dictionary.items()}
                self.dp_shard_dictionary.append(new_sh_dict)
   
        else:
            # generate shards and expected minibatch runtime
            self.shard_dictionary, self.shard_to_input_dict, self.shard_to_output_dict, self.batch_time = self.partitioner.shard(
                                                                            self.layer_dictionary, 
                                                                            next(iter(self._old_data)), 
                                                                            buffer, 
                                                                            self.lr, 
                                                                            self.verbose
                                                                            )
        
        
        
        print("SHARING MODEL MEMORY")
        for key, shard in self.shard_dictionary.items():
            shard.model = shard.model.cpu()
            shard.model.share_memory()
            
        print("CREATING EXECUTION PROCESS")
        self.base_model_process = mp.Process(target=update_shared_parameters, args=(self.shard_dictionary, self.mp_queue))
        self.base_model_process.start()
            
        self.total_shards = self.shard_dictionary.keys() # which shards to run overall
            
        for d in range(self.data_parallel_degree):
            self.get_new_batch(d)
        
    def get_new_batch(self, dp_instance):
        
        
        """
            DP Synchronization
        
        """

        # Reset the tensor dictionary and gradient dictionary
        self.dp_tensor_dictionary[dp_instance] = {}
        self.dp_candidate_shards[dp_instance] = set()
        self.dp_completed_shards[dp_instance] = set()
        self.dp_grad_dictionary[dp_instance] = {}
        self.dp_blocked_shards[dp_instance] = set()
        
        
        
        for key in self.endpoint_keys:
            self.dp_grad_dictionary[dp_instance][key] = None

        
        """
        
            Batch statistics update
        
        """
        
        # Update minibatch, handle new batch as well
        try:
            batch, label = next(self.dataloader) # get the next minibatch
        except StopIteration:
            del self.dataloader # delete old dataloader
            self.epochs -= 1 # count off an epoch
            self.dataloader = iter(self._old_data) # regenerate the dataloader
            self.total_length = len(self._old_data)
            self.minibatches_remaining = len(self._old_data) # set the number of minibatches
            batch, label = next(self.dataloader) # get the next minibatch

        self.minibatches_remaining -= 1 # decrement minibatch count
        
        completed_mbs = self.total_length - self.minibatches_remaining
        if (completed_mbs % 1 == 0):
            print("MODEL: {}, EPOCH: {}, MBS: {} / {}".format(self.name, self.total_epochs-self.epochs, completed_mbs, self.total_length))
            
            
        """
            Re-initialize tensor dictionary
            
        """
        
        # initialize the tensor dictionary with initial minibatch
        if not isinstance(batch, torch.Tensor):
            for idx, tensor in enumerate(batch):
                self.dp_tensor_dictionary[dp_instance]["batch_{}".format(idx)] = tensor
        else:
            self.dp_tensor_dictionary[dp_instance]["batch_0"] = batch 

        if not isinstance(label, torch.Tensor):
            for idx, tensor in enumerate(label):
                self.dp_tensor_dictionary[dp_instance]["label_{}".format(idx)] = tensor
        else:
            self.dp_tensor_dictionary[dp_instance]["label_0"] = label 
        
        self.update_task(dp_instance)
        
            
    """
        Updates the ModelTask. Should be called after each shard completion.
    """
    def update_task(self, dp_instance, completed_index=None, ret_tensor_dictionary=None, ret_grad_dictionary=None):

        if completed_index is not None:
            if self.shard_dictionary[completed_index].direction == "b":
                self.mp_queue.put(completed_index)
                self.mp_queue.put(self.dp_shard_dictionary[dp_instance][completed_index].state_dict())
                
            self.dp_completed_shards[dp_instance].add(completed_index)
            
        if ret_tensor_dictionary is not None:
            self.dp_tensor_dictionary[dp_instance].update(ret_tensor_dictionary)
            
        if ret_grad_dictionary is not None:
            self.dp_grad_dictionary[dp_instance].update(ret_grad_dictionary)
        
        # get shards that are not already executed or primed
        eval_shards = (self.total_shards - self.dp_completed_shards[dp_instance]) - self.dp_candidate_shards[dp_instance]  - self.dp_blocked_shards[dp_instance]
        
        # check if their dependencies have been resolved
        for shard in eval_shards:
            if self.shard_dictionary[shard].direction == "f":
                req_tensors = self.shard_to_input_dict[shard]
                if all(req in self.dp_tensor_dictionary[dp_instance] for req in req_tensors):
                    self.dp_candidate_shards[dp_instance].add(shard)
            else:
                greq_tensors = self.shard_to_output_dict[shard]
                req_tensors = self.shard_to_input_dict[shard]
                if all(req in self.dp_tensor_dictionary[dp_instance] and greq in self.dp_grad_dictionary[dp_instance] for req, greq in zip(req_tensors, greq_tensors)):
                    self.dp_candidate_shards[dp_instance].add(shard)
        
        
        print("TASK {} DP INSTANCE {} OUTPUTS UPDATED ON COMPLETION OF {}".format(self.name, dp_instance, completed_index))
        
        if (len(self.dp_completed_shards[dp_instance]) == len(self.total_shards)):
            self.get_new_batch(dp_instance)
            
            
    """
        "Planned" candidates after shard key completes. Doesn't affect any class variables.
    """
    def get_expected_update(self, key, dp_instance):
        
        gen_shard = self.shard_dictionary[key]
        f_expected_total, b_expected_total = set(self.dp_tensor_dictionary[dp_instance].keys()), set(self.dp_grad_dictionary[dp_instance].keys())
        
        if gen_shard.direction == "f":
            expected_outs = self.shard_to_output_dict[key] # the intermediates the active shard will spit out
            f_expected_total.update(expected_outs)
        else:
            expected_outs = self.shard_to_input_dict[key] # the gradients the active shard will spit out
            b_expected_total.update(expected_outs)
            
        # Non-candidate shards that need to be evaluated for inclusion
        eval_shards = (self.total_shards - self.dp_completed_shards[dp_instance]) - self.dp_candidate_shards[dp_instance]  - self.dp_blocked_shards[dp_instance]
            
        possibles = copy.copy(self.dp_candidate_shards[dp_instance])
        
        
        for shard in eval_shards:
            if self.shard_dictionary[shard].direction == "f":
                req_tensors = self.shard_to_input_dict[shard] # for this shard to run we need xyz
                if all(req in f_expected_total for req in req_tensors):
                    possibles.add(shard)
            else:
                greq_tensors = self.shard_to_output_dict[shard] # for this shard to run we need xyz
                req_tensors = self.shard_to_input_dict[shard]
                if all(req in f_expected_total and greq in b_expected_total for req, greq in zip(req_tensors, greq_tensors)):
                    possibles.add(shard)
                    
        return possibles
    
    
    """
        Will be called for buffering. Blacklists this shard from being selected in future scheduling until minibatch
        finishes.
    """
    
    def get_shard(self, key, dp_instance):
        self.dp_blocked_shards[dp_instance].add(key)
        self.dp_candidate_shards[dp_instance].discard(key)
        shard_task = self.dp_shard_dictionary[dp_instance][key]
        return shard_task
    
    def get_shard_blind_from_dp(self, dp_instance):
        key = self.dp_candidate_shards[dp_instance].pop()
        self.dp_blocked_shards[dp_instance].add(key)
        shard_task = self.dp_shard_dictionary[dp_instance][key]
        
        return key, shard_task
    

    
    def get_available_shard_inputs(self, key, dp_instance):
        shard_task = self.shard_dictionary[key]
        tensor_requests = self.shard_to_input_dict[key]
        
        input_tensors = [k for k in tensor_requests if k in self.dp_tensor_dictionary[dp_instance]]
        if shard_task.direction == "b":
            grad_requests = self.shard_to_output_dict[key]
            grad_tensors = [k for k in grad_requests if k in self.dp_grad_dictionary[dp_instance]]
        else:
            grad_tensors = None
        return input_tensors, grad_tensors
    
    def get_shard_inputs(self, key, dp_instance):
        shard_task = self.shard_dictionary[key]
        tensor_requests = self.shard_to_input_dict[key]
        print("SEARCHING FOR INPUTS FOR SHARD {} DP INSTANCE {}".format(key, dp_instance))
        print("AVAILABLE TENSORS: {}".format(self.dp_tensor_dictionary[dp_instance].keys()))
        input_tensors = {k: self.dp_tensor_dictionary[dp_instance][k] for k in tensor_requests}
        if shard_task.direction == "b":
            grad_requests = self.shard_to_output_dict[key]
            grad_tensors = {k: self.dp_grad_dictionary[dp_instance][k] for k in grad_requests}
        else:
            grad_tensors = None
        return input_tensors, grad_tensors