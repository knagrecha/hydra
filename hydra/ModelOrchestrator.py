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
from hydra.components.logger import Logger

from datetime import datetime
import concurrent.futures
import sys
import signal
from torch import multiprocessing
import traceback

global_timer = timer()
thread_lock = threading.Lock()



class ModelOrchestrator():
    def __init__(self, tasks, verbose=0, buffer=None):
        available_gpus = min(torch.cuda.device_count(), len(tasks))
        
        
        self.all_devices, self.available_devices = list(range(available_gpus)), list(range(available_gpus))
        
        self.active_devices = []
        self.verbose = verbose
        self.tasks = copy.copy(tasks) # copy to avoid modifying user input
        
        for i in self.tasks:
            i.global_timer = global_timer
            
        self.idle_tasks = copy.copy(tasks)
        self.active_tasks, self.cached_tasks = [], []
        
        self.buffer = buffer
        self.sleep_event = threading.Event()
        
        self.thread_pool = concurrent.futures.ThreadPoolExecutor()
   
    def setup_all_models(self):
        start = timer()
        for task in self.tasks:
            task.setup(self.verbose, self.buffer)
        end = timer()
        
        
    def generate(self):
        self.setup_all_models()
        if self.verbose == 1:
            self.logger = Logger(self.tasks)

    """
        Setups the parameters for the train_shard function
    """

    def train_shard_on_device(self, chosen_task, chosen_shard, chosen):
        try:
            profile_timer_start = timer()
            device = torch.device("cuda:{0}".format(chosen))
            if chosen_shard.idx == 0:
                if chosen_shard.direction == "f":
                    chosen_task.get_new_batch()

            # defining the parameters
            criterion, back_input, batch, labels = None, None, None, None

            # FORWARD PASS
            if chosen_shard.direction == "f":
                batch = chosen_task.saved_inter_output[-1] # get the 'front' of our saved intermediates
                
                # FORWARD + LOSS (last shard)
                if chosen_shard.idx == len(chosen_task.forward_shards)- 1:
                    arg_list = [batch, chosen_task.label, chosen_task.criterion, device, chosen_task.scaler]
                    chosen_task.scaler, new_batch, chosen_task.last_loss = chosen_shard.run(arg_list)
                    
                # REGULAR FORWARD
                else:
                    arg_list = [batch, device]
                    new_batch = chosen_shard.run(arg_list)
                    
                    # Detach data for forward passes.
                    if not isinstance(new_batch, torch.Tensor):
                        new_batch = [i.detach_() for i in new_batch]
                    else:
                        new_batch = new_batch.detach()
                        
            # BACKWARD PASS       
            else:
                batch = chosen_task.gradient
                back_input = chosen_task.saved_inter_output.pop() # the "checkpoint" input
                arg_list = [batch, device, back_input, chosen_task.scaler]
                chosen_task.scaler, new_batch = chosen_shard.run(arg_list)

           
            # if not cached, or if it is cached but needs to be swapped out anyway due to it being a multi-shard model, return to CPU
            if (chosen_task not in self.cached_tasks or (chosen_task.queue_len > 1 and chosen_shard.idx != 0)):
                chosen_shard.model = chosen_shard.model.to("cpu", non_blocking=True)
            

            # if the next shard of this model is on GPU, try to skip CPU offload for the output
            if (new_batch is not None):
                if (chosen_task not in self.cached_tasks or chosen_task.queue_len == 1):
                    new_batch = move_batch_to_device(new_batch, "cpu")
               
            # if Backward pass, or the ForwardLoss pass, update the gradient we are chaining backwards
            if chosen_shard.direction == "b" or (chosen_shard.idx == len(chosen_task.forward_shards)- 1 and chosen_shard.direction == "f"):
                chosen_task.gradient = new_batch

            # if forward, prep it for next pass.
            else:
                chosen_task.saved_inter_output.append(new_batch)

            if len(chosen_task.queue) == 0: # if the model queue wasn't refreshed, it implies the model has finished training.
                self.tasks.remove(chosen_task)
                chosen_task.cleanup() # remove for production
                print("Task {} has finished at time {}.".format(chosen_task.name, timer()-global_timer))
            else:
                self.idle_tasks.append(chosen_task)
                chosen_task.clear_settings()

            self.unlock_device(chosen)
            self.active_tasks.remove(chosen_task)
               
            profile_timer_end = timer()

            chosen_shard.time_cost = profile_timer_end - profile_timer_start # update model running times to make profiler more accurate
            chosen_task.new_total_time += chosen_shard.time_cost
            self.sleep_event.set()
            
        except Exception as e:
            traceback.print_exc()
            print(e)
            
    def lock_device(self, chosen):
        self.active_devices.append(chosen)
        self.available_devices.remove(chosen)
    
    def unlock_device(self, chosen):
        self.active_devices.remove(chosen)
        self.available_devices.append(chosen)
    
   
    def train_models(self):
        print("TRAINING STARTS")
        
        ctr = 0
        cache_task, cache_device = None, None # use this to try and "guess" the next task's completion time for that shard.

        self.cached_tasks, running_tasks = [None for x in range(len(self.all_devices))], [None for x in range(len(self.all_devices))]

        global thread_lock
        # initial run

        temp_active = []
        for chosen_device in self.all_devices:
            
            # LRT scheduler
            task_times = [(i.total_time * i.batches_remaining) + i.total_length * i.total_time * i.epochs for i in self.idle_tasks]
            chosen_task = self.idle_tasks[np.argmax(task_times)] 

            # scheduler metadata around the task
            self.lock_device(chosen_device)
            chosen_task.setup_timing(chosen_device)
            self.active_tasks.append(chosen_task)
            running_tasks[chosen_device] = chosen_task
            
            # get the shard to train
            chosen_shard = chosen_task.get_shard()
            temp_active.append(chosen_device)
            self.idle_tasks.remove(chosen_task)
            
            
            self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)

        # calculate shards to cache
        candidate_tasks = self.idle_tasks[:]
        for active_device in temp_active:
            
            # give preference to the task already on device
            active_task = running_tasks[active_device]
            if (len(active_task.queue) > 0):
                cache_task = active_task
                lrt = active_task.total_time * active_task.batches_remaining + active_task.total_length * active_task.total_time * active_task.epochs + active_task.total_length
            else:
                lrt = -1
                cache_task = None
                
            # calculate LRT
            for i in candidate_tasks:
                if (len(i.queue) > 0):
                    task_time = (i.total_time * i.batches_remaining ) + i.total_length * i.total_time * i.epochs
                    if task_time > lrt:
                        lrt = task_time
                        cache_task = i
                        
            # once a cached task has been selected, prefetch it to GPU and remove it from the LRT caching candidate pool            
            if cache_task is not None:
                if (cache_task.queue_len > 1):
                    cache_task.queue[0].model.to("cuda:{0}".format(active_device), non_blocking=True)
                    if cache_task != active_task:
                        candidate_tasks.remove(cache_task)
                    self.cached_tasks[active_device] = cache_task
                
        
        # main scheduling loop
        start = timer()
        while len(self.tasks) > 0:
            try:
                self.sleep_event.wait()
                self.sleep_event.clear()
                if self.verbose == 1:
                    self.logger.refresh()
                if (len(self.tasks) == 0):
                    break
                    
                temp_active = []
                holder = self.available_devices[:] # avoid iterating over available devices, which is modified inside the loop
                for chosen_device in holder:
                    # get the cached task for that device
                    chosen_task = self.cached_tasks[chosen_device]
                    if chosen_task is not None:
                        
                        self.idle_tasks.remove(chosen_task)
                        self.lock_device(chosen_device)
                        chosen_task.setup_timing(chosen_device)
                        self.active_tasks.append(chosen_task)
                        
                        chosen_shard = chosen_task.get_shard()
                        running_tasks[chosen_device] = chosen_task
                        temp_active.append(chosen_device)
                        
                        # clean caching metadata
                        self.cached_tasks[chosen_device] = None
                        self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)

                candidate_tasks = [x for x in self.idle_tasks if x not in self.cached_tasks]
                
                # replace the cached tasks with new shards
                for active_device in temp_active:
                    
                    # give preference to the task on device
                    active_task = running_tasks[active_device]
                    if (len(active_task.queue) > 0):
                        cache_task = active_task
                        lrt = active_task.total_time * active_task.batches_remaining + active_task.total_length * active_task.total_time * (active_task.epochs - 1)
                    else:
                        lrt = -1
                        cache_task = None
                        
                    # calculate LRT
                    for i in candidate_tasks:
                        if (len(i.queue) > 0):
                            task_time = (i.total_time * i.batches_remaining ) + i.total_length * i.total_time * (i.epochs - 1)

                            if task_time > lrt:
                                lrt = task_time
                                cache_task = i
                                
                    # cache task
                    if cache_task is not None:
                        cache_task.queue[0].model.to("cuda:{0}".format(active_device), non_blocking=True)
                        if cache_task != active_task:
                            candidate_tasks.remove(cache_task)
                    self.cached_tasks[active_device] = cache_task

            except KeyboardInterrupt:
                if thread_lock.locked():
                    thread_lock.release()
                print ("Caught KeyboardInterrupt, terminating workers")
                if self.verbose == 1:
                    self.logger.cleanup()
                end = timer()
                print()
                print()
                print("TOTAL TIME TAKEN: {}".format(end - start))
                sys.exit(0)
                
        end = timer()
        print("TOTAL TIME TAKEN: {}".format(end - start))
        if self.verbose == 1:
            self.logger.cleanup()
        return (end-start)
        
