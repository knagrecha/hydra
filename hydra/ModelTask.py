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
import threading
from collections import defaultdict
import copy







def update_shared_parameters(base_model, pipe):
    
    local_optimizers = {k: torch.optim.SGD(base_model[k].model.parameters(), lr=0.0001) for k in base_model.keys()}
    while True:
        dp_instance = pipe.recv()
        shard_key = pipe.recv()
        grad_dict = pipe.recv()
        #print("SHARD {} UPDATE STARTED".format(shard_key))
        if grad_dict is not None:
            for name, param_x in base_model[shard_key].model.named_parameters():
                param_x.grad = grad_dict[name]
            local_optimizers[shard_key].step()
            del grad_dict
        base_model[shard_key].model.zero_grad()
        new_shard = base_model[shard_key].copy()
        pipe.send(new_shard)
        pipe.send(shard_key)
        pipe.send(dp_instance)
        

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
        mp.set_start_method('spawn', force=True)

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
        self.dp_multithreading_events = []
        
        
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
        
        self.mp_parent, self.mp_child = mp.Pipe()
        
    def listen_to_pipe(self, pipe):
        while True:
            model = pipe.recv()
            key = pipe.recv()
            dp_instance = pipe.recv()
            self.dp_shard_dictionary[dp_instance][key] = model
            self.dp_multithreading_events[dp_instance][key].set()
            #print("SHARD {} UPDATE FINISHED".format(key))
        
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
                #print("CREATING DP INSTANCE: {}".format(d))
                new_sh_dict = {k: task.copy() for k, task in shard_dictionary.items()}
                new_sh_event_dict = {k: threading.Event() for k in shard_dictionary.keys()}
                for k in new_sh_event_dict.keys():
                    new_sh_event_dict[k].set()
                self.dp_shard_dictionary.append(new_sh_dict)
                self.dp_multithreading_events.append(new_sh_event_dict)
   
        else:
            # generate shards and expected minibatch runtime
            self.shard_dictionary, self.shard_to_input_dict, self.shard_to_output_dict, self.batch_time = self.partitioner.shard(
                                                                            self.layer_dictionary, 
                                                                            next(iter(self._old_data)), 
                                                                            buffer, 
                                                                            self.lr, 
                                                                            self.verbose
                                                                            )

            
        #print("CREATING PARAMETER UPDATE PROCESS")
        self.base_model_process = mp.Process(target=update_shared_parameters, args=(self.shard_dictionary, self.mp_child))
        self.base_model_process.start()
        
        #print("CREATING LISTENER PROCESS")
        self.listener = threading.Thread(target=self.listen_to_pipe, args=(self.mp_parent, ))
        self.listener.start()
            
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
        
        
        
        # Wait for all parameter updates in this batch to finish
        st = timer()
        for key, value in self.dp_multithreading_events[dp_instance].items():
            value.wait()
        end = timer()
        print("SPENT {} on sync".format(end-st))
        self.update_task(dp_instance)
        
            
    """
        Updates the ModelTask. Should be called after each shard completion.
    """
    def update_task(self, dp_instance, grad_dict=None, completed_index=None, ret_tensor_dictionary=None, ret_grad_dictionary=None):
        
        #print("UPDATE TASK STARTED")
        if completed_index is not None:
            #print("SENDING TO MP")
            self.mp_parent.send(dp_instance)
            self.mp_parent.send(completed_index)
            self.mp_parent.send(grad_dict)

            self.dp_completed_shards[dp_instance].add(completed_index)
            
        #print("CONTINUING TO EXECUTE NORMALLY")
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
                    if self.dp_multithreading_events[dp_instance][shard].is_set():
                        self.dp_candidate_shards[dp_instance].add(shard)
            else:
                greq_tensors = self.shard_to_output_dict[shard]
                req_tensors = self.shard_to_input_dict[shard]
                if all(req in self.dp_tensor_dictionary[dp_instance] and greq in self.dp_grad_dictionary[dp_instance] for req, greq in zip(req_tensors, greq_tensors)):
                    self.dp_candidate_shards[dp_instance].add(shard)

        if (len(self.dp_completed_shards[dp_instance]) == len(self.total_shards)):
            self.get_new_batch(dp_instance)
        
        #print("FINISHED UPDATE")
            
            

    def get_shard_blind_from_dp(self, dp_instance):
        key = self.dp_candidate_shards[dp_instance].pop()
        self.dp_blocked_shards[dp_instance].add(key)
        shard_task = self.dp_shard_dictionary[dp_instance][key]
        self.dp_multithreading_events[dp_instance][key].clear()
        return key, shard_task

    
    def get_shard_inputs(self, key, dp_instance):
        shard_task = self.shard_dictionary[key]
        tensor_requests = self.shard_to_input_dict[key]
        input_tensors = {k: self.dp_tensor_dictionary[dp_instance][k] for k in tensor_requests}
        if shard_task.direction == "b":
            grad_requests = self.shard_to_output_dict[key]
            grad_tensors = {k: self.dp_grad_dictionary[dp_instance][k] for k in grad_requests}
        else:
            grad_tensors = None
        return input_tensors, grad_tensors