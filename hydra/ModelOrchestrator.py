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
            
        self.active_tasks = {k: (None, None) for k in self.all_devices} # currently active (task, shard) tuple by device
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

    def train_shard_on_device(self, model_shard, model_task, input_tensors, grad_tensors, chosen_device, chosen_shard_index):
        returned_tensors = model_shard.run(chosen_device, input_tensors, grad_tensors) # run the model
        ret_keys = returned_tensors.keys()
        mark_saved = set()
        if returned_tensors:
            do_save = False
            for key in ret_keys:
                if returned_tensors[key] is not None:
                    returned_tensors[key] = returned_tensors[key].to("cpu", non_blocking=True)

        if model_shard.direction == "f":
            model_task.update_task(chosen_shard_index, ret_tensor_dictionary=returned_tensors)
        else:
            model_task.update_task(chosen_shard_index, ret_grad_dictionary=returned_tensors)

        if model_task.epochs <= 0:
            self.tasks.remove(model_task)
        else:
            self.idle_tasks.append(model_task)

        self.unlock_device(chosen_device)
        self.active_tasks[chosen_device] = (None, None)

        self.sleep_event.set()
            

    def lock_device(self, chosen):
        self.active_devices.append(chosen)
        self.available_devices.remove(chosen)
    
    def unlock_device(self, chosen):
        self.active_devices.remove(chosen)
        self.available_devices.append(chosen)
    
   
    def train_models(self):
        print("****************************TRAINING STARTS***************************************")
        global thread_lock
        for device in self.available_devices:
            candidate_tasks = [t for t in self.tasks if len(t.candidate_shards) > 0] # selection candidates
            if (len(candidate_tasks) > 0):

                task_times = [(i.mini_batch_time * i.minibatches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i in candidate_tasks]
                chosen_task = candidate_tasks[np.argmax(task_times)]
                self.lock_device(device)
                chosen_key, chosen_shard = chosen_task.get_shard_blind() # get the first candidate 
                in_tensors, grad_tensors = chosen_task.get_shard_inputs(chosen_key)
                self.active_tasks[device] = (chosen_task, chosen_key)
                candidate_tasks.remove(chosen_task)
                self.thread_pool.submit(self.train_shard_on_device, chosen_shard, chosen_task, 
                                        in_tensors, grad_tensors, device, chosen_key)
        while len(self.tasks) > 0:

            try:
                self.sleep_event.wait()
                self.sleep_event.clear()

                if (len(self.tasks) == 0):
                    break
                # for each free device
                for device in self.available_devices:
                    candidate_tasks = [t for t in self.tasks if len(t.candidate_shards) > 0] # selection candidates
                    if (len(candidate_tasks) > 0):

                        task_times = [(i.mini_batch_time * i.minibatches_remaining) + (i.mini_batch_time * i.total_length * i.epochs) for i in candidate_tasks]
                        chosen_task = candidate_tasks[np.argmax(task_times)]
                        self.lock_device(device)
                        chosen_key, chosen_shard = chosen_task.get_shard_blind() # get the first candidate 
                        in_tensors, grad_tensors = chosen_task.get_shard_inputs(chosen_key)
                        self.active_tasks[device] = (chosen_task, chosen_key)
                        candidate_tasks.remove(chosen_task)
                        self.thread_pool.submit(self.train_shard_on_device, chosen_shard, chosen_task, 
                                                in_tensors, grad_tensors, device, chosen_key)

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
        
