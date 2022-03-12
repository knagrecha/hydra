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
        available_gpus = 
        self.all_devices = [ torch.device("cuda:{}".format(idx)) for idx in range(torch.cuda.device_count()) ]
        self.available_devices = copy.copy(self.all_devices)
        self.active_devices = []
        self.tasks = copy.copy(tasks)
        self.idle_tasks = copy.copy(tasks)
        for i in self.tasks:
            i.global_timer = global_timer
        self.active_tasks = []
        self.cached_tasks = []
        self.sleep_event = threading.Event()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor()

        
    def generate(self):
        self.setup_all_models()

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

    def train_shard_on_device(self, model_shard, model_task, input_tensors, grad_tensors, chosen_device, chosen_shard_index):
        try:
            profile_timer_start = timer()

            
            returned_tensors = model_shard.run(chosen_device, input_tensors, grad_tensors) # run the model
            
            # if any of the tensors are requested by a cached shard, then do not move them (mark_saved), otherwise
            # send to CPU
            ret_keys = returned_tensors.keys()
            mark_saved = {}
            if returned_tensors:
                do_save = False
                for cached_task, cached_shard in self.cached_shard:
                    if cached_task == model_task:
                        for key in ret_keys:
                            if key in cached_task.shard_to_input_dict[cached_shard]:
                                mark_saved.add(key)
                
                for key in ret_keys:
                    if key not in mark_saved:
                        returned_tensors[key] = returned_tensors[key].to("cpu", non_blocking=True)

            
            if model_shard.direction == "f":
                model_task.update_task(chosen_shard_index, ret_tensor_dictionary=returned_tensors)
            else:
                model_task.update_task(chosen_shard_index, ret_grad_dictionary=returned_tensors)
                
            if model_task.epochs <= 0:
                self.tasks.remove(model_task)
                print("Task {} has finished at time {}.".format(chosen_task.name, timer()-global_timer))
            else:
                self.idle_tasks.append(model_task)

            self.unlock_device(chosen_device)
            self.active_tasks.remove(model_task)

            profile_timer_end = timer()
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
        print("****************************TRAINING STARTS***************************************")
        cache_task = None # use this to try and "guess" the next task's completion time for that shard.
        cache_device = None
        self.cached_tasks = [None for x in range(len(self.all_devices))]
        running_tasks = {}
        global thread_lock
        
        
        # select initial tasks
        for chosen_device in self.all_devices:
            task_times = [(i.mini_batch_time * i.batches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i in self.idle_tasks]
            chosen_task = self.idle_tasks[np.argmax(task_times)]
            self.lock_device(chosen_device)
            self.active_tasks.append(chosen_task)
            running_tasks[chosen_device] = chosen_task
            chosen_shard = chosen_task.get_shard()
            
            self.idle_tasks.remove(chosen_task)
            self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)
        #print(self.active_devices)

        if CACHE_SYSTEM:
         # recalculate cached shards
            considerables = self.idle_tasks.copy()
            for active_device in self.active_devices:
                active_task = running_tasks[active_device]
                if (len(active_task.queue) > 0):
                    #print("active task does NOT finish this cycle")
                    considerables.append(active_task)

                lrt = -1
                cache_task = None
                #print("Considering {}".format(considerables))
                for i in considerables:
                    if (len(i.queue) > 0):
                        task_time = (i.total_time * i.batches_remaining ) + i.total_length * i.total_time * i.epochs
                        if task_time > lrt:
                            lrt = task_time
                            cache_task = i
                    
                if cache_task is not None:
                    if (cache_task.queue_len > 1):
                        cache_task.queue[0].model.to("cuda:{0}".format(active_device), non_blocking=True)
                    considerables.remove(cache_task)
                    self.cached_tasks[active_device] = cache_task
                
                if (active_task) in considerables:
                    considerables.remove(active_task)

        
        #print(cached_tasks)
        
        # runtime
        start = timer()
        old_time = 0
        while len(self.tasks) > 0:
            if (self.verbose == 1 and timer() - old_time > 10):
                for task in self.tasks:
                    print(task.name + ": Epoch {}, {} / {} minibatches complete, remaining time (approx.): {:.2f}hrs, last runtime: {:.2f}, last loss: {:.2f} | ".format( task.total_epochs - task.epochs, task.total_length - task.batches_remaining, task.total_length, task.remaining_runtime/3600, task.last_runtime, task.last_loss))
                old_time = timer()
            try:
                self.sleep_event.wait()
                self.sleep_event.clear()
                if (len(self.tasks) == 0):
                    break
   
                if CACHE_SYSTEM and len(self.idle_tasks) > 0:
                    
                    holder = self.available_devices.copy()
                    #print(holder)
                    temp_active = []
                    for chosen_device in holder:
                        chosen_task = self.cached_tasks[chosen_device]
                        if chosen_task is not None:
                            self.idle_tasks.remove(chosen_task)
                            self.lock_device(chosen_device)
                            chosen_task.setup_timing(chosen_device)
                            self.active_tasks.append(chosen_task)
                            chosen_shard = chosen_task.get_shard()
                            running_tasks[chosen_device] = chosen_task
                            temp_active.append(chosen_device)
                            #print("Training {} on {}".format(chosen_task.name, chosen_device))
                            self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)

                    # recalculate cached shards
                    considerables = [x for x in self.idle_tasks if x not in self.cached_tasks]
                    
                    #print("Devices needing a cache {}".format(temp_active))
                    for active_device in temp_active:
                        
                        active_task = running_tasks[active_device]
                        cache_possibles = considerables.copy()
                        if (len(active_task.queue) > 0):
                            cache_possibles.append(active_task)
                        lrt = -1
                        cache_task = None
                        #print("Considering {} for device {}".format([task.name for task in cache_possibles], active_device))
                        if (len(cache_possibles) == 0):
                            self.cached_tasks[active_device] = None
                    
                        for i in cache_possibles:
                            #print(len(i.queue))
                            if (len(i.queue) > 0):
                                task_time = (i.total_time * i.batches_remaining ) + i.total_length * i.total_time * i.epochs
                                i.remaining_runtime = task_time
                                if task_time > lrt:
                                    lrt = task_time
                                    cache_task = i
                        #print("CACHING {} to {}".format(cache_task.name, active_device))
                        if cache_task is not None:
                            


                            if (cache_task.queue_len > 1 and cache_task.my_device != active_device):
                                cache_task.queue[0].model.to("cuda:{0}".format(active_device), non_blocking=True)

                            self.cached_tasks[active_device] = cache_task
                            if cache_task in considerables:
                                considerables.remove(cache_task)
                        else:
                            self.cached_tasks[active_device] = None
                    
                else:
                    
                    holder = self.available_devices.copy()
                    for chosen_device in holder:
                        lrt = -1
                        #print(self.idle_tasks)
                        if (len(self.idle_tasks) != 0):
                            for i in self.idle_tasks:
                                task_time = (i.total_time * i.batches_remaining) + i.total_length * i.total_time * i.epochs
                                if task_time > lrt:
                                    lrt = task_time
                                    chosen_task = i

                            self.idle_tasks.remove(chosen_task)
                            self.lock_device(chosen_device)
                            chosen_task.setup_timing(chosen_device)
                            self.active_tasks.append(chosen_task)
                            chosen_shard = chosen_task.get_shard()
                            #print("Training {} on {}".format(chosen_task.name, chosen_device))
                            self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)
                #thread_lock.release()
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
        
