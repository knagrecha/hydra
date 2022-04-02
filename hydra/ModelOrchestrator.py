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
import copy
from torch.autograd import grad
import torch.optim as optim
import threading
import gc
from timeit import default_timer as timer
from hydra.utilities import get_free_space, move_batch_to_device, get_used_space
import math
#import curses
import numpy as np
from .ModelTask import ModelTask
from datetime import datetime
import concurrent.futures
import sys
import signal
from torch import multiprocessing as mp
import traceback
global_timer = timer()
thread_lock = threading.Lock()

"""

    TODO:
        Avoid sending through pipe CPU tensors that are already buffered in a local process (local_saved_x_tensors)
        Have GenericExecutor only send back relevant tensors, not all of them.

"""



"""

    Multiprocessed execution that "owns" a particular device. Waits for pipe input then executes and returns.
    
    The communication procedure:
    
        1. Main process (scheduler) sends out execution details
        2. Executor sends gradient dictionary (or None). Marks end of execution
        3. Scheduler send back to the executor two flags, one telling the executor whether or not to offload the model
           and a second telling it which tensors to offload.
        4. Executor offloads what it needs to and sends back the tensors.
    
"""

def execute_train(device_rank, pipe):
    local_saved_f_tensors = {} # held-in place during execution. Use the process-local copies.
    local_saved_b_tensors = {}
    print("STARTED EXECUTION")
    while True:
        # block until inputs received
        print("WAITING")
        model_shard = pipe.recv()
        input_tensors = pipe.recv()
        grad_tensors = pipe.recv()
        chosen_shard_index = pipe.recv()
        
        input_tensors.update(local_saved_f_tensors)
        if grad_tensors is not None:
            grad_tensors.update(local_saved_b_tensors)
        
        local_saved_f_tensors = {}
        local_saved_b_tensors = {}

        print("RECEIVED")
        returned_tensors = model_shard.run(device_rank, input_tensors, grad_tensors) # run the model
        
        # execute
        if model_shard.direction == "b":
            grad_dict = {}
            for key, value in model_shard.model.named_parameters():
                grad_dict[key] = value.grad.cpu().clone()
            # send back the data
            pipe.send(grad_dict)
        else:
            pipe.send(None)
            
        should_offload = pipe.recv() # should I offload my model?
        if should_offload:
            model_shard.model.to("cpu", non_blocking=True)
            
        f_save_tensors = pipe.recv() # which forward tensors (if any) should I not offload?
        b_save_tensors = pipe.recv() # which backward tensors (if any) should I not offload?
        ret_keys = returned_tensors.keys()
        print("PRODUCED KEYS: {}".format(ret_keys))
        
        for key in ret_keys:
            if key in f_save_tensors:
                local_saved_f_tensors[key] = returned_tensors[key]
            elif  model_shard.direction == "b" and key in b_save_tensors:
                local_saved_b_tensors[key] = returned_tensors[key]

            # Only send back CPU tensors, avoids CUDA sharing errors.    
            if returned_tensors[key] is not None:
                returned_tensors[key] = returned_tensors[key].to("cpu", non_blocking=True)
                

        pipe.send(returned_tensors) # send back the tensors


"""
    The "orchestrator"/scheduler that assigns ModelTasks for training.

"""

class ModelOrchestrator():
    def __init__(self, tasks):
        torch.multiprocessing.set_start_method('spawn', force=True)
        self.all_devices = [ torch.device("cuda:{}".format(idx)) for idx in range(torch.cuda.device_count()) ] # all devices
        self.available_devices = copy.copy(self.all_devices) # "empty" devices
        self.active_devices = [] # currently active devices
        
        self.send_pipes = []
        self.rec_pipes = []
        for x in range(len(self.all_devices)):
            parent, child = mp.Pipe()
            self.send_pipes.append(parent)
            self.rec_pipes.append(child)
            
        print("CREATING PROCESSES")
        self.mp_processes = [mp.Process(target=execute_train, args=(x, self.rec_pipes[x])) for x in range(len(self.all_devices))]
        for proc in self.mp_processes:
            proc.start()
        print("PROCESSES STARTED")
        
        self.tasks = copy.copy(tasks) # list of tasks
        self.idle_tasks = copy.copy(tasks) # list of idle tasks
        
        # setup basal timers
        for i in self.tasks:
            i.global_timer = global_timer 
            
        self.active_tasks = {k: (None, None) for k in self.all_devices} # currently active (task, shard) tuple by device
        # currently cached (task, shard, in_tensors, grad_tensors, key) by 
        self.cached_tasks = {k: (None, None, None) for k in self.all_devices}
        self.sleep_event = threading.Event()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor()
        self.verbose = 1


    def log(self, message):
        if self.verbose == 1:
            screen_lock.acquire()
            print(message)
            screen_lock.release()



    """
        Executes training. 
        Receives as input a ShardTask, the ModelTask that owns it, 
        input tensors, grad tensors (can be None), and an assigned device.
    """

    def train_shard_on_device(self, pipe, model_shard, model_task, input_tensors, grad_tensors, chosen_shard_index, chosen_device):
        print("EXECUTION STARTS")
        try:
            pipe.send(model_shard)
            pipe.send(input_tensors)
            pipe.send(grad_tensors)
            pipe.send(chosen_shard_index)
        except Exception as e:
            print(e)

        print("SENT TO PIPE")
        
        grad_dict = pipe.recv()
        print("RECEVIED GRAD DICT")
        
        cached_task, cached_shard, shard_key = self.cached_tasks[chosen_device]
        if cached_task == model_task:
            if cached_task.shard_dictionary[shard_key].model != model_shard.model:
                pipe.send(True)
            else:
                pipe.send(False)
            
            
            f_savables = cached_task.shard_to_input_dict[shard_key]
            print("SAVE INTERMEDIATES: {}".format(f_savables))
            pipe.send(f_savables) # save items cached on same device
            
            if cached_shard.direction == "b":
                b_savables = cached_task.shard_to_output_dict[shard_key]
                print("SAVE GRADIENTS: {}".format(b_savables))
                pipe.send(b_savables)
            else:
                pipe.send(None)

        else:
            pipe.send(set()) # don't save any
                
        print("SENT OFFLOAD INFO")
        
        ret_tensors = pipe.recv()
        print("RECEIVED TENSORS")
        try:
            if model_shard.direction == "f":
                model_task.update_task(chosen_shard_index, grad_dict, ret_tensor_dictionary=ret_tensors)
            else:
                model_task.update_task(chosen_shard_index, grad_dict, ret_grad_dictionary=ret_tensors)

            if model_task.epochs <= 0:
                self.tasks.remove(model_task)
            else:
                self.idle_tasks.append(model_task)

            self.unlock_device(chosen_device)
            self.active_tasks[chosen_device] = (None, None)
            self.sleep_event.set()
        except Exception as e:
            print(e)

            
    def lock_device(self, chosen):
        self.active_devices.append(chosen)
        self.available_devices.remove(chosen)
    
    def unlock_device(self, chosen):
        self.active_devices.remove(chosen)
        self.available_devices.append(chosen)
    
   
    def train_models(self):
        print("****************************TRAINING STARTS***************************************")
        global thread_lock
        # select initial tasks
        for device in self.all_devices:
            candidate_tasks = [t for t in self.tasks if len(t.candidate_shards) > 0] # selection candidates
            if len(candidate_tasks) > 0:
                task_times = [(i.mini_batch_time * i.minibatches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i in candidate_tasks]
                chosen_task = candidate_tasks[np.argmax(task_times)]
                self.lock_device(device)
                # TODO: Introduce some kind of actual selection process
                chosen_key, chosen_shard = chosen_task.get_shard_blind() # get the first candidate 
                in_tensors, grad_tensors = chosen_task.get_shard_inputs(chosen_key)
                self.active_tasks[device] = (chosen_task, chosen_key)
                print("SUBMITTED TO PROCESS {}".format(device.index))
                self.thread_pool.submit(self.train_shard_on_device, self.send_pipes[device.index], chosen_shard, chosen_task, in_tensors, grad_tensors, chosen_key, device)

        for device in self.all_devices:
            # Build initial buffers
            candidate_tasks = []
            active_task_specific_shard_pool = None # shards of same task that will be valid after current shard
            for t in self.tasks:
                if t == self.active_tasks[device][0]:
                    active_task_specific_shard_pool = t.get_expected_update(self.active_tasks[device][1])
                    if len(active_task_specific_shard_pool) > 0:
                        candidate_tasks.append(t) # selection candidates
                else:
                    if len(t.candidate_shards) > 0:
                        candidate_tasks.append(t) # selection candidates

            lrt = -1 # Sharded-LRTF selection - start by calculating LRT
            cache_task = None
            for candidate in candidate_tasks:
                task_time = ((candidate.mini_batch_time * candidate.minibatches_remaining) + 
                                 (candidate.mini_batch_time * candidate.total_length * candidate.epochs))
                if task_time > lrt:
                    lrt = task_time
                    cache_task = candidate

            if cache_task is not None:
                if cache_task == self.active_tasks[device][0]:
                    chosen_key = active_task_specific_shard_pool.pop()
                    chosen_shard = cache_task.get_shard(chosen_key)
                else:
                    chosen_key, chosen_shard = cache_task.get_shard_blind() # get the first candidate 
                    
                self.cached_tasks[device] = (cache_task, chosen_shard, chosen_key)
                chosen_shard.model = chosen_shard.model.to(device, non_blocking=True) # Buffer up the model

            start = timer()

        while len(self.tasks) > 0:

            try:
                self.sleep_event.wait()
                self.sleep_event.clear()

                if (len(self.tasks) == 0):
                    break
                # for each free device
                for device in self.available_devices:
                    # trigger buffered tasks if available
                    cache_task, chosen_shard, chosen_key = self.cached_tasks[device]
                    if cache_task is not None:
                        in_tensors, grad_tensors = cache_task.get_shard_inputs(chosen_key)
                        if cache_task is not None:
                            self.idle_tasks.remove(cache_task)
                            self.lock_device(device)
                            self.active_tasks[device] = (cache_task, chosen_key)
                            self.cached_tasks[device] = None, None, None
                            print("SUBMITTED TO PROCESS {}".format(device.index))
                            self.thread_pool.submit(self.train_shard_on_device, self.send_pipes[device.index], chosen_shard, cache_task, in_tensors, grad_tensors, chosen_key, device)
                    
                    # if no cached task was possible, revert to standard scheduling
                    else:
                        candidate_tasks = [t for t in self.tasks if len(t.candidate_shards) > 0] # selection candidates
                        if (len(candidate_tasks) > 0):
                            
                            task_times = [(i.mini_batch_time * i.minibatches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i in candidate_tasks]
                            chosen_task = candidate_tasks[np.argmax(task_times)]
                            self.lock_device(device)
                            chosen_key, chosen_shard = chosen_task.get_shard_blind() # get the first candidate 
                            in_tensors, grad_tensors = chosen_task.get_shard_inputs(chosen_key)
                            self.active_tasks[device] = (chosen_task, chosen_key)
                            candidate_tasks.remove(chosen_task)
                            print("SUBMITTED TO PROCESS {}".format(device.index))
                            self.thread_pool.submit(self.train_shard_on_device, self.send_pipes[device.index], chosen_shard, chosen_task, in_tensors, grad_tensors, chosen_key, device)

                        
                    # Replace buffers for this device
                    candidate_tasks = []
                    active_task_specific_shard_pool = None # shards of same task that will be valid after current shard
                    for t in self.tasks:
                        if t == self.active_tasks[device][0]:
                            active_task_specific_shard_pool = t.get_expected_update(self.active_tasks[device][1])
                           
                            if len(active_task_specific_shard_pool) > 0:
                                candidate_tasks.append(t) # selection candidates
                        else:
                            if len(t.candidate_shards) > 0:
                                candidate_tasks.append(t) # selection candidates
                                
                    lrt = -1 # Sharded-LRTF selection - start by calculating LRT
                    cache_task = None
                    for candidate in candidate_tasks:
                        task_time = ((candidate.mini_batch_time * candidate.minibatches_remaining) + 
                                         (candidate.mini_batch_time * candidate.total_length * candidate.epochs))
                        if task_time > lrt:
                            lrt = task_time
                            cache_task = candidate

                    if cache_task is not None:
                        # if caching from same model
                        if cache_task == self.active_tasks[device][0]:
                            chosen_key = active_task_specific_shard_pool.pop()
                            chosen_shard = cache_task.get_shard(chosen_key)
                        else:
                            chosen_key, chosen_shard = cache_task.get_shard_blind() # get the first candidate 
                            
                            
                        #print("DB'ing {} at {}".format(chosen_key, timer()))
                        self.cached_tasks[device] = (cache_task, chosen_shard, chosen_key)
                        chosen_shard.model = chosen_shard.model.to(device, non_blocking=True) # Buffer up the model
                        
                        available_in_tensors, available_grad_tensors = cache_task.get_available_shard_inputs(chosen_key)
                        for t in available_in_tensors:
                            cache_task.tensor_dictionary[t] = cache_task.tensor_dictionary[t].to(device, non_blocking=True)
                        if available_grad_tensors is not None:
                            for t in available_grad_tensors:
                                if cache_task.grad_dictionary[t] is not None:
                                    cache_task.grad_dictionary[t] = cache_task.grad_dictionary[t].to(device, non_blocking=True)
                        #print("FINISHED DB'ing at {}".format(timer()))
     

            except KeyboardInterrupt:
                if thread_lock.locked():
                    thread_lock.release()
                
                while len(self.send_pipes) > 0:
                    del self.send_pipes[0]
                    del self.rec_pipes[0]

                for process in self.mp_processes:
                    process.terminate()
                    #process.close()
                    
                print ("Caught KeyboardInterrupt, terminating workers")
                end = timer()
                print()
                print()
                print("TOTAL TIME TAKEN: {}".format(end - start))


                sys.exit(0)

            
     
        end = timer()
        print("TOTAL TIME TAKEN: {}".format(end - start))
        
