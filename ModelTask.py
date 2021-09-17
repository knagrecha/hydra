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
from hydra.nn.utilities import get_free_space
import numpy as np
from torch import multiprocessing as multiprocessing



def get_load_time(shard, a, device):
    b = a[:-1]
    b = [d.to(device) for d in b]
    shard.model.to(device)
    for idx, i in enumerate(b):
        b[idx] = i.cpu()
    for idx, i in enumerate(b):
        b[idx] = i.to(device)
    for idx, i in enumerate(b):
        b[idx] = i.cpu()
    while (len(b) > 0):
        del b[0]
    del b
    del a
    torch.cuda.empty_cache()

class ModelTask():
    def __init__(self, name, model, criterion, dataloader, lr, epochs, global_timer=None, use_scaler=False):
        self.global_timer = global_timer
        self.name = name
        self.model = model
        self.lr = lr
        self.total_epochs = epochs
        self.epochs = epochs
        self.criterion = criterion
        self.batch_size = dataloader.batch_size
        
        self._old_data = dataloader
        self.dataloader = iter(dataloader)
        self.total_length = len(self._old_data)
        self.queue = []
        self.batches_remaining = len(dataloader)
        self.saved_inter_output = []
        self.gradient = None

        self.wastage = 0
        if (use_scaler):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self.label = None
        self.batch_time = 0
        
        self.queue_len = 0
        self.curr_cycle = 0
        self.last_mini_time = 0
        self.list_of_waste = []
      
        self.active_time = 0
        self.my_device = None
        self.anticipated_curr_shard_time = 0
        self.setup_complete = False

    def clear(self):
        del self._old_data
        del self.dataloader
        del self.queue
        del self.label
        
    def cleanup(self):
        for shard in self.model.f_shards:
            del shard
        del self.model.layers
        del self.model
        while(len(self.saved_inter_output) > 0):
            del self.saved_inter_output[0]
        del self.dataloader
        del self._old_data
        del self.scaler
        while(len(self.queue) > 0):
            del self.queue[0]
        if (self.gradient is not None):
            del self.gradient
        
        
    def clear_settings(self):
        self.anticipated_curr_shard_time = 0
        
    def manual_setup(self, split_indices):
        self.model.setup_manual(self.criterion, next(self.dataloader), split_indices)
        self.setup_complete = True
        
    def setup(self, verbose, buffer):
        if (not self.setup_complete):
            self.model.verbose = verbose
            self.model.setup(self.criterion, next(self.dataloader), buffer)
        
        
        start = timer()
        self.dataloader = iter(self._old_data)
        a = next(self.dataloader)
        self.batch_time = timer() - start
        del self.dataloader
        self.dataloader = iter(self._old_data)
        
        
        available_gpus = torch.cuda.device_count()
        available_devices = list(range(available_gpus))
        free_spaces = [get_free_space(x) for x in available_devices]
        device_idx = np.argmin(free_spaces)
        device = torch.device("cuda:"+str(device_idx))
    
        self.queue.extend(self.model.f_shards)
        self.queue.extend(self.model.b_shards)

        for shard in self.queue:
            torch.cuda.empty_cache()
            start = timer()
            get_load_time(shard, a, device)
            self.list_of_waste.append(timer() - start)
            shard.model = shard.model.cpu()
        print(self.list_of_waste)
        
        self.queue_len = len(self.queue)
     
    def setup_timing(self, device):
        
        if (next(self.queue[0].model.parameters()).device == torch.device(device)):
            #print(next(self.queue[0].model.parameters()).device)
            self.anticipated_curr_shard_time = timer() - self.global_timer + self.queue[0].time_cost
            self.my_device = device
        else:
            self.anticipated_curr_shard_time = timer() - self.global_timer + self.queue[0].time_cost + self.list_of_waste[self.curr_cycle]
            self.my_device = device
            
    def get_new_batch(self):
        print("Minibatch time taken: {}".format(timer()-self.last_mini_time))
        
        try:
            batch_full = next(self.dataloader)
        except StopIteration:
            del self.dataloader
            
            self.epochs -= 1
            print("Task {} finished an epoch, {} remaining".format(self.name, self.epochs))
            self.dataloader = iter(self._old_data)
            self.batches_remaining = len(self._old_data)
            batch_full = next(self.dataloader)

        self.batches_remaining -= 1
        
        if ((self.total_length / self.batches_remaining) % 5 == 0):
            print ("[Task {}: {} / {} batches complete]".format(self.name, self.total_length - self.batches_remaining, self.total_length))
        
        self.saved_inter_output.append(batch_full[0:len(batch_full)-1])
        self.label = batch_full[-1]
        
        self.curr_cycle = 0
        
        
        self.last_mini_time = timer()
    def get_shard(self):
        shard = self.queue.pop(0)
        if (self.epochs > 1 or self.batches_remaining >= 1):
            self.queue.append(shard)
        self.curr_cycle +=1
        return shard
