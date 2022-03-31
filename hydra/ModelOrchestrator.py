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
from torch import multiprocessing
import traceback
global_timer = timer()
thread_lock = threading.Lock()

"""
    The "orchestrator"/scheduler that assigns ModelTasks for training.

"""


class ModelOrchestrator():
    def __init__(self, tasks):
        self.all_devices = [ torch.device("cuda:{}".format(idx)) for idx in range(torch.cuda.device_count()) ] # all devices
        self.available_devices = copy.copy(self.all_devices) # "empty" devices
        self.active_devices = [] # currently active devices
        self.tasks = copy.copy(tasks) # list of tasks
        self.idle_tasks = copy.copy(tasks) # list of idle tasks
        
        # setup basal timers
        for i in self.tasks:
            i.global_timer = global_timer 
            
        # currently active (task, shard, dp_instance) tuple by device
        self.active_tasks = {k: (None, None, None) for k in self.all_devices} 
        # currently cached (task, shard, in_tensors, grad_tensors, key, dp_instance) by 
        self.cached_tasks = {k: (None, None, None, None) for k in self.all_devices}
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
        
        Also receives shard metadata.
    """

    def train_shard_on_device(self, model_shard, model_task, input_tensors, grad_tensors, chosen_device, chosen_shard_index, chosen_dp_instance):
        try:
            returned_tensors = model_shard.run(chosen_device, input_tensors, grad_tensors) # run the model
            #end = timer()
            # if any of the tensors are requested by a cached shard, then do not move them (mark_saved), otherwise
            # send to CPU
            
            
            ret_keys = returned_tensors.keys()
            mark_saved = set()
            if returned_tensors:
                do_save = False
                
                """
                    TODO: If we run into bugs, maybe restrict this evaluation to self.cached_tasks[chosen_device]
                
                """
                
                for device, (cached_task, cached_shard, shard_key, dp_instance) in self.cached_tasks.items():
                    if cached_task == model_task:
                        if cached_task.dp_shard_dictionary[dp_instance][shard_key].model != model_shard.model:
                            model_shard.model.to("cpu", non_blocking=True)
                        
                        if chosen_dp_instance == dp_instance:
                            savables = cached_task.shard_to_input_dict[shard_key]
                            for key in ret_keys:
                                if key in savables:
                                    mark_saved.add(key)

                for key in ret_keys:
                    if key not in mark_saved:
                        if returned_tensors[key] is not None:
                            returned_tensors[key] = returned_tensors[key].to("cpu", non_blocking=True)

            

            if model_shard.direction == "f":
                model_task.update_task(chosen_dp_instance, chosen_shard_index, ret_tensor_dictionary=returned_tensors)
            else:
                model_task.update_task(chosen_dp_instance, chosen_shard_index, ret_grad_dictionary=returned_tensors)
                
            if model_task.epochs <= 0:
                self.tasks.remove(model_task)
            else:
                self.idle_tasks.append(model_task)

            self.unlock_device(chosen_device)
            self.active_tasks[chosen_device] = (None, None, None)

            self.sleep_event.set()
            
        except Exception as e:
            traceback.print_exc()
            print(e)
            
    def lock_device(self, chosen):
        print("LOCKING {}".format(chosen))
        self.active_devices.append(chosen)
        self.available_devices.remove(chosen)
    
    def unlock_device(self, chosen):
        print("UNLOCKING {}".format(chosen))
        self.active_devices.remove(chosen)
        self.available_devices.append(chosen)
    
   
    def train_models(self):
        print("****************************TRAINING STARTS***************************************")
        global thread_lock
        
        # select initial tasks
        for device in self.all_devices:
            candidate_tasks = []
            for t in self.tasks:
                for d in range(t.data_parallel_degree):
                    if len(t.dp_candidate_shards[d]) > 0:
                        candidate_tasks.append((t, d))

            if len(candidate_tasks) > 0:
                task_times = [(i.mini_batch_time * i.minibatches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i, j in candidate_tasks]
                chosen_task, chosen_dp = candidate_tasks[np.argmax(task_times)]
                self.lock_device(device)
                
                """
                    Blindly select a candidate shard from the task and get its initial inputs as well as metadata.
                    
                """
                
                # TODO: Introduce some kind of actual selection process
                chosen_key, chosen_shard = chosen_task.get_shard_blind_from_dp(chosen_dp) # get the first candidate 
                in_tensors, grad_tensors = chosen_task.get_shard_inputs(chosen_key, chosen_dp)
                
                self.active_tasks[device] = (chosen_task, chosen_key, chosen_dp)
                print("EXECUTING TASK {} SHARD {} DP {}".format(chosen_task.name, chosen_key, chosen_dp))
                self.thread_pool.submit(self.train_shard_on_device, chosen_shard, chosen_task, 
                                        in_tensors, grad_tensors, device, chosen_key, chosen_dp)

        """        
            Initial DB pass.
        """
        
        for device in self.all_devices:
            # Build initial buffers
            candidate_tasks = []
            active_task_specific_shard_pool = None # shards of same task that will be valid after current shard
            
            for t in self.tasks:
                for d in range(t.data_parallel_degree):
                    if t == self.active_tasks[device][0] and d == self.active_tasks[device][2]:
                        # what outputs will be available AFTER the current shard completes?
                        active_task_specific_shard_pool = t.get_expected_update(self.active_tasks[device][1], self.active_tasks[device][2])

                        if len(active_task_specific_shard_pool) > 0:
                            candidate_tasks.append((t, d)) # selection candidates
                    else:
                        if len(t.dp_candidate_shards[d]) > 0:
                            candidate_tasks.append((t, d)) # selection candidates

            print("CANDIDATE TASKS FOR DEVICE {}: {}".format(device, candidate_tasks))
            lrt = -1 # Sharded-LRTF selection - start by calculating LRT
            cache_task = None
            chosen_dp = 0
            for candidate, dp_instance in candidate_tasks:
                task_time = ((candidate.mini_batch_time * candidate.minibatches_remaining) + 
                                 (candidate.mini_batch_time * candidate.total_length * candidate.epochs))
                if task_time > lrt:
                    lrt = task_time
                    cache_task = candidate
                    chosen_dp = dp_instance

            
            # Get a shard from the chosen task
            if cache_task is not None:
                
                # if caching the same task as the one on-device
                if cache_task == self.active_tasks[device][0]:
                    chosen_key = active_task_specific_shard_pool.pop()
                    chosen_shard = cache_task.get_shard(chosen_key, chosen_dp)
                else:
                    chosen_key, chosen_shard = cache_task.get_shard_blind_from_dp(chosen_dp) # get the first candidate 
                    
                self.cached_tasks[device] = (cache_task, chosen_shard, chosen_key, chosen_dp)
                print("BUFFERING TASK {} SHARD {} DP {}".format(cache_task.name, chosen_key, chosen_dp))
                chosen_shard.model = chosen_shard.model.to(device, non_blocking=True) # Buffer up the model

        start = timer()

        # standard scheduling
        while len(self.tasks) > 0:

            try:
                self.sleep_event.wait()
                self.sleep_event.clear()

                if (len(self.tasks) == 0):
                    break
                    
                # for each free device
                for device in self.available_devices:
                    # trigger buffered tasks if available
                    cache_task, chosen_shard, chosen_key, chosen_dp = self.cached_tasks[device]
                    if cache_task is not None:
                        in_tensors, grad_tensors = cache_task.get_shard_inputs(chosen_key, chosen_dp)
                        if cache_task is not None:
                            self.idle_tasks.remove(cache_task)
                            self.lock_device(device)
                            self.active_tasks[device] = (cache_task, chosen_key, chosen_dp)
                            self.cached_tasks[device] = (None, None, None, None)
                            print("EXECUTING TASK {} SHARD {} DP {}".format(cache_task.name, chosen_key, chosen_dp))
                            self.thread_pool.submit(self.train_shard_on_device, chosen_shard, cache_task, 
                                        in_tensors, grad_tensors, device, chosen_key, chosen_dp)
                    
                    # if no cached task was possible, revert to standard scheduling
                    else:
                        candidate_tasks = []
                        for t in self.tasks:
                            for d in range(t.data_parallel_degree):
                                if len(t.dp_candidate_shards[d]) > 0:
                                    candidate_tasks.append((t, d))
                                        
                        if (len(candidate_tasks) > 0):
                            
                            task_times = [(i.mini_batch_time * i.minibatches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i, j in candidate_tasks]
                            
                            chosen_task, chosen_dp = candidate_tasks[np.argmax(task_times)]
                            self.lock_device(device)
                            chosen_key, chosen_shard = chosen_task.get_shard_blind_from_dp(chosen_dp) # get the first candidate 
                            in_tensors, grad_tensors = chosen_task.get_shard_inputs(chosen_key, chosen_dp)
                            self.active_tasks[device] = (chosen_task, chosen_key, chosen_dp)
                            self.thread_pool.submit(self.train_shard_on_device, chosen_shard, chosen_task, 
                                                    in_tensors, grad_tensors, device, chosen_key, chosen_dp)

                        
                    # Replace buffers for this device
                    candidate_tasks = []
                    active_task_specific_shard_pool = None # shards of same task that will be valid after current shard
                    for t in self.tasks:
                        for d in range(t.data_parallel_degree):
                            if t == self.active_tasks[device][0] and d  == self.active_tasks[device][2]:
                                active_task_specific_shard_pool = t.get_expected_update(self.active_tasks[device][1], self.active_tasks[device][2])

                                if len(active_task_specific_shard_pool) > 0:
                                    candidate_tasks.append((t, d)) # selection candidates
                            else:
                                if len(t.candidate_shards) > 0:
                                    candidate_tasks.append((t, d)) # selection candidates

                    lrt = -1 # Sharded-LRTF selection - start by calculating LRT
                    cache_task = None
                    chosen_dp = 0
                    for candidate, dp_instance in candidate_tasks:
                        task_time = ((candidate.mini_batch_time * candidate.minibatches_remaining) + 
                                         (candidate.mini_batch_time * candidate.total_length * candidate.epochs))
                        if task_time > lrt:
                            lrt = task_time
                            cache_task = candidate
                            chosen_dp = dp_instance

                    if cache_task is not None:
                        # if caching from same model
                        if cache_task == self.active_tasks[device][0]:
                            chosen_key = active_task_specific_shard_pool.pop()
                            chosen_shard = cache_task.get_shard(chosen_key, chosen_dp)
                        else:
                            chosen_key, chosen_shard = cache_task.get_shard_blind_from_dp(chosen_dp) # get the first candidate 
                            
                            
                        #print("DB'ing {} at {}".format(chosen_key, timer()))
                        self.cached_tasks[device] = (cache_task, chosen_shard, chosen_key, chosen_dp)
                        chosen_shard.model = chosen_shard.model.to(device, non_blocking=True) # Buffer up the model
                        print("BUFFERING TASK {} SHARD {} DP {}".format(cache_task.name, chosen_key, chosen_dp))
                        
                        
                        # POSSIBLY COMMENT OUT THIS LOWER SECTION
                        available_in_tensors, available_grad_tensors = cache_task.get_available_shard_inputs(chosen_key, chosen_dp)
                        for t in available_in_tensors:
                            cache_task.dp_tensor_dictionary[chosen_dp][t] = cache_task.dp_tensor_dictionary[dp_instance][t].to(device, non_blocking=True)
                        if available_grad_tensors is not None:
                            for t in available_grad_tensors:
                                if cache_task.dp_grad_dictionary[dp_instance][t] is not None:
                                    cache_task.dp_grad_dictionary[dp_instance][t] = cache_task.dp_grad_dictionary[dp_instance][t].to(device, non_blocking=True)
                        #print("FINISHED DB'ing at {}".format(timer()))
     

            except KeyboardInterrupt:
                if thread_lock.locked():
                    thread_lock.release()


                print ("Caught KeyboardInterrupt, terminating workers")
                end = timer()
                print()
                print()
                print("TOTAL TIME TAKEN: {}".format(end - start))


                sys.exit(0)

            
     
        end = timer()
        print("TOTAL TIME TAKEN: {}".format(end - start))
        
