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



class ModelOrchestrator():
    def __init__(self, tasks):
        
        #multiprocessing.set_start_method('spawn', force=True)

        available_gpus = min(torch.cuda.device_count(), len(tasks))
        
       
        
        self.all_devices = list(range(available_gpus))
        self.available_devices = list(range(available_gpus))
        self.active_devices = []
        self.verbose = 0
        self.tasks = copy.copy(tasks)
        for i in self.tasks:
            i.global_timer = global_timer
        self.idle_tasks = copy.copy(tasks)
        self.active_tasks = []
        self.cached_tasks = []
        self.buffer = None
        self.sleep_event = threading.Event()
        #self.process_pool = multiprocessing.Pool(processes = cpus, maxtasksperchild=1)
        
        #print("Creating Thread pool.")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor()
        #print("Pool created.")
   
    def setup_all_models(self):
        start = timer()
        for task in self.tasks:
            task.setup(self.verbose, self.buffer)
            #print("TASK {} finished setup".format(task.name))
        #print("ALL TASKS SETUP!")

            
        end = timer()
        
    def generate(self):
        self.setup_all_models()

    def log(self, message):
        if self.verbose == 1:
            screen_lock.acquire()
            print(message)
            screen_lock.release()



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
                batch = chosen_task.saved_inter_output[-1]
                
                # FINAL FORWARD
                if chosen_shard.idx == len(chosen_task.forward_shards)- 1:
                    arg_list = [batch, chosen_task.label, chosen_task.criterion, device, chosen_task.scaler]
                    chosen_task.scaler, new_batch, chosen_task.last_loss = chosen_shard.run(arg_list)
                # REGULAR FORWARD
                else:
                    arg_list = [batch, device]
                    new_batch = chosen_shard.run(arg_list)
                    
                    # Detach data for forward passes. Back and final return list-type gradients which don't need
                    # to be detached anyway.
                    if not isinstance(new_batch, torch.Tensor):
                        new_batch = [i.detach_() for i in new_batch]
                    else:
                        new_batch = new_batch.detach()
            # BACKWARD PASS       
            else:
                batch = chosen_task.gradient
                back_input = chosen_task.saved_inter_output[-1]
                arg_list = [batch, device, back_input, chosen_task.scaler]
                chosen_task.scaler, new_batch = chosen_shard.run(arg_list)

           
            if (chosen_task not in self.cached_tasks or (chosen_task.queue_len > 1 and chosen_shard.idx != 0)):
                chosen_shard.model = chosen_shard.model.to("cpu", non_blocking=True)
            

            # Hold in place if possible
            if (new_batch is not None):
                if (chosen_task not in self.cached_tasks or chosen_task.queue_len == 1):
                    new_batch = move_batch_to_device(new_batch, "cpu")
               
            my_batch = new_batch

            l_f = False
            if chosen_shard.idx == len(chosen_task.forward_shards)- 1 and chosen_shard.direction == "f":
                l_f = True

            # if backward pass, update the gradient
            if chosen_shard.direction == "b" or l_f:
                chosen_task.gradient = my_batch
                chosen_task.saved_inter_output.pop()

            # if forward, prep it for next pass.
            else:
                chosen_task.saved_inter_output.append(my_batch)

            #thread_lock.acquire()
            if len(chosen_task.queue) == 0:
                self.tasks.remove(chosen_task)
                chosen_task.cleanup() # remove for production
                print("Task {} has finished at time {}.".format(chosen_task.name, timer()-global_timer))
            else:
                self.idle_tasks.append(chosen_task)
                chosen_task.clear_settings()

            self.unlock_device(chosen)
            self.active_tasks.remove(chosen_task)
               
            
            

            profile_timer_end = timer()

            chosen_shard.time_cost = profile_timer_end - profile_timer_start # time for process running, assuming model is on device.
            #thread_lock.release()
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
        cache_task = None # use this to try and "guess" the next task's completion time for that shard.
        cache_device = None
        
        self.cached_tasks = [None for x in range(len(self.all_devices))]
        running_tasks = [None for x in range(len(self.all_devices))]
        
        global thread_lock
        # initial run

        temp_active = []
        for chosen_device in self.all_devices:
            task_times = [(i.total_time * i.batches_remaining) + i.total_length * i.total_time * i.epochs for i in self.idle_tasks]
            chosen_task = self.idle_tasks[np.argmax(task_times)]

            self.lock_device(chosen_device)
            chosen_task.setup_timing(chosen_device)
            self.active_tasks.append(chosen_task)
            running_tasks[chosen_device] = chosen_task
            chosen_shard = chosen_task.get_shard()
            temp_active.append(chosen_device)
            self.idle_tasks.remove(chosen_task)
            self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)
        #print(self.active_devices)
        # recalculate cached shards
        considerables = self.idle_tasks[:]
        for active_device in temp_active:
            active_task = running_tasks[active_device]
            if (len(active_task.queue) > 0):
                cache_task = active_task
                lrt = active_task.total_time * active_task.batches_remaining + active_task.total_length * active_task.total_time * active_task.epochs + active_task.total_length
            else:
                lrt = -1
                cache_task = None
            for i in considerables:
                if (len(i.queue) > 0):
                    task_time = (i.total_time * i.batches_remaining ) + i.total_length * i.total_time * i.epochs
                    if task_time > lrt:
                        lrt = task_time
                        cache_task = i
            if cache_task is not None:
                if (cache_task.queue_len > 1):
                    cache_task.queue[0].model.to("cuda:{0}".format(active_device), non_blocking=True)
                    if cache_task != active_task:
                        considerables.remove(cache_task)
                    self.cached_tasks[active_device] = cache_task
                
        
        # runtime
        start = timer()
        while len(self.tasks) > 0:
            try:
                self.sleep_event.wait()
                self.sleep_event.clear()
                if (len(self.tasks) == 0):
                    break
                    
                temp_active = []
                holder = self.available_devices[:] # avoid iterating over available devices, which is modified inside the loop
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
                        self.cached_tasks[chosen_device] = None
                        self.thread_pool.submit(self.train_shard_on_device, chosen_task, chosen_shard, chosen_device)

                considerables = [x for x in self.idle_tasks if x not in self.cached_tasks]
                for active_device in temp_active:
                    active_task = running_tasks[active_device]
                    if (len(active_task.queue) > 0):
                        cache_task = active_task
                        lrt = active_task.total_time * active_task.batches_remaining + active_task.total_length * active_task.total_time * active_task.epochs + active_task.total_length
                    else:
                        lrt = -1
                        cache_task = None
                    for i in considerables:
                        if (len(i.queue) > 0):
                            task_time = (i.total_time * i.batches_remaining ) + i.total_length * i.total_time * i.epochs
                            if task_time > lrt:
                                lrt = task_time
                                cache_task = i
                    if cache_task is not None:
                        cache_task.queue[0].model.to("cuda:{0}".format(active_device), non_blocking=True)
                        if cache_task != active_task:
                            considerables.remove(cache_task)
                    self.cached_tasks[active_device] = cache_task

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
        return (end-start)
        
