import torch
import copy
from torch.autograd import grad
import torch.optim as optim
import threading
import gc
from timeit import default_timer as timer
from mptorch.nn.utilities import get_free_space, get_used_space
import math
#import curses
import numpy as np
from .ModelTask import ModelTask
from datetime import datetime
import concurrent.futures
import sys
import signal
from torch import multiprocessing

global_timer = timer()
thread_lock = threading.Lock()




def train_shard(shard, batch_input, device, labels=None, criterion=None, lr=None, back_input=None, scaler=None, optimizer = None):

    if shard.direction == "f":
        old = next(shard.model.parameters()).device
        shard.model.to(device, non_blocking=True)

        if not isinstance(batch_input, torch.Tensor):
            batch_input = [x.to(device, non_blocking=True) for x in batch_input]
        else:
            batch_input = batch_input.to(device, non_blocking=True)

        if labels != None:
            shard.model.zero_grad()  # zeroes the gradient buffers of all parameters
            optimizer.zero_grad()  # zero the gradient buffers
            if not isinstance(labels, torch.Tensor):
                labels = [x.to(device, non_blocking=True) for x in labels]
            else:
                labels = labels.to(device, non_blocking=True)

            if (shard.idx != 0):
                if not isinstance(batch_input, torch.Tensor):
                    for batch in batch_input:
                        batch.requires_grad_(True)
                else:
                    batch_input.requires_grad_(True)


            with torch.cuda.amp.autocast():
                ns_labels = shard.model(batch_input)
                loss = criterion(ns_labels, labels)

            loss = scaler.scale(loss)

            loss.backward()


            pass_back_gradients = []
            if (shard.idx != 0):
                # pass_back_gradients are on device
                if not isinstance(batch_input, torch.Tensor):
                    pass_back_gradients = [ batch.grad for batch in batch_input ]
                else:
                    pass_back_gradients.append(batch_input.grad)
            else:
                pass_back_gradients = None

            scaler.step(optimizer)
            scaler.update()
            del optimizer


            shard.model.zero_grad()

            #shard_model = shard.model.to("cpu", non_blocking=True)

            del labels

            if not isinstance(batch_input, torch.Tensor):
                while (len(batch_input) > 0):
                    del batch_input[0]
            else:
                del batch_input
            return scaler, pass_back_gradients



        else:

            with torch.no_grad() and torch.cuda.amp.autocast():
                ns_labels = shard.model(batch_input)

            #shard_model = shard.model.to("cpu", non_blocking=True)

            if not isinstance(batch_input, torch.Tensor):
                while (len(batch_input) > 0):
                    del batch_input[0]
                del batch_input
            else:
                del batch_input
            return ns_labels


    # backpropagation
    else:

        shard.model.to(device, non_blocking=True)
        shard.model.zero_grad()  # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()  # zero the gradient buffers

        if not isinstance(back_input, torch.Tensor):
            #print("Back input is a list")
            toy_input = [x.to(device, non_blocking=True) for x in back_input]

            if (shard.idx != 0):
                for m_input in toy_input:
                    #print(m_input)
                    if isinstance(m_input, torch.Tensor):
                        m_input.requires_grad_(True)

        else:
            toy_input = back_input.to(device, non_blocking=True)
            if (shard.idx != 0):
                toy_input.requires_grad_(True)     

        if not isinstance(batch_input, torch.Tensor):
            batch_input = [x.to(device, non_blocking=True) for x in batch_input]
        else:
            batch_input = batch_input.to(device, non_blocking=True) 


        with torch.cuda.amp.autocast():
            toy_output = shard.model(toy_input)

        toy_output = scaler.scale(toy_output)
        torch.autograd.backward(toy_output, batch_input)
        del toy_output
        del batch_input
        pass_back_gradients = None
        if shard.idx != 0: # the first backwards pass need not compute back pass gradients.
            if (not isinstance(toy_input, torch.Tensor)):
                pass_back_gradients = [i.grad for i in toy_input]
                for m_input in toy_input:
                    m_input.requires_grad_(False)
            else:
                pass_back_gradients = toy_input.grad
                toy_input.requires_grad_(False)
            # the user will pass in what WAS the input for this stage!
        scaler.step(optimizer)
        scaler.update()


        if not isinstance(toy_input, torch.Tensor):
            while (len(toy_input) > 0):
                del toy_input[0]
            del toy_input
        else:
            del toy_input

        del optimizer

        shard.model.zero_grad()
        #print("This [TRAIN STEP] took {} seconds".format(timer()-st))

        #shard.model = shard.model.to("cpu", non_blocking=True)

        return scaler, pass_back_gradients



class ModelOrchestrator():
    def __init__(self, tasks, cpus):
        
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
        
        print("Creating Thread pool.")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor()
        print("Pool created.")
   
    def setup_all_models(self):
        start = timer()
        for task in self.tasks:
            
            task.setup(self.verbose, self.buffer)
            print("TASK {} finished setup".format(task.name))
        print("ALL TASKS SETUP!")

            
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
        #print("starting train for {}".format(chosen_task.name))
        profile_timer_start = timer()
        device = torch.device("cuda:{0}".format(chosen))
        if chosen_shard.idx == 0:
            if chosen_shard.direction == "f":
                #print("Getting new batch for {}, {} batches remaining".format(chosen_task.name, chosen_task.batches_remaining))
                chosen_task.get_new_batch()



        # defining the parameters
        criterion, back_input, batch, labels, optimizer = None, None, None, None, None

        # if its a forward pass, the batch is the latest item in the intermediate outputs
        if chosen_shard.direction == "f":
            batch = chosen_task.saved_inter_output[-1]

            # if its the last forward pass (and hence the first backward pass), also send a label and criterion.
            if chosen_shard.idx == len(chosen_task.model.f_shards)- 1:
                labels = chosen_task.label
                criterion = chosen_task.criterion
                optimizer = chosen_shard.optimizer
        else:
            batch = chosen_task.gradient
            back_input = chosen_task.saved_inter_output[-1]
            optimizer = chosen_shard.optimizer



        start = timer()

        if (chosen_shard.direction == "b" or chosen_shard.idx == len(chosen_task.model.f_shards) - 1):
            chosen_task.scaler, new_batch = train_shard(chosen_shard, batch, device, labels, criterion, chosen_task.lr, back_input, chosen_task.scaler, optimizer)

            if (new_batch is not None):
                if (chosen_task not in self.cached_tasks or chosen_task.queue_len == 1):

                    if not isinstance(new_batch, torch.Tensor):
                        new_batch = [i.to("cpu", non_blocking=True) for i in new_batch]
                    else:
                        new_batch = new_batch.to("cpu", non_blocking=True)


        # FORWARD
        else:
            new_batch = train_shard(chosen_shard, batch, device, labels, criterion, chosen_task.lr, back_input, chosen_task.scaler)
            if (chosen_task not in self.cached_tasks or chosen_task.queue_len == 1):
                if not isinstance(new_batch, torch.Tensor):
                    new_batch = [i.detach().to("cpu", non_blocking=True) for i in new_batch]
                else:
                    new_batch = new_batch.detach().to("cpu", non_blocking=True)
            else:
                if not isinstance(new_batch, torch.Tensor):
                    new_batch = [i.detach_() for i in new_batch]
                else:
                    new_batch = new_batch.detach_()
        if (chosen_task not in self.cached_tasks or chosen_task.queue_len > 1):
            chosen_shard.model = chosen_shard.model.to("cpu", non_blocking=True)


        #print("Train for {} completed.".format(chosen_task.name))
        end = timer()
        chosen_task.wastage = end - start


        #print("Batch transfers")
        if new_batch is not None:
            if isinstance(new_batch, torch.Tensor):
                my_batch = new_batch.cpu()
            else:
                #print(new_batch)
                my_batch = [x.cpu() for x in new_batch]
            del new_batch
        else:
            my_batch = None

        l_f = False
        if chosen_shard.idx == len(chosen_task.model.f_shards)- 1 and chosen_shard.direction == "f":
            l_f = True

        # if backward pass, update the gradient
        if chosen_shard.direction == "b" or l_f:
            #print("Handling backward gradients")
            chosen_task.gradient = my_batch
            chosen_task.saved_inter_output.pop()

        # if forward, prep it for next pass.
        else:
            chosen_task.saved_inter_output.append(my_batch)

        #thread_lock.acquire()
        if len(chosen_task.queue) == 0:
            #print("TASK FULLY COMPLETE")
            self.tasks.remove(chosen_task)
            print("Task {} has completely finished at time {}.".format(chosen_task.name, timer()-global_timer))
            chosen_task.cleanup() # remove for production
        else:
            chosen_task.clear_settings()
            self.idle_tasks.append(chosen_task)

        #print("Task {} has finished at time {}.".format(chosen_task.name, timer()-global_timer))
        self.unlock_device(chosen)
        self.active_tasks.remove(chosen_task)

        profile_timer_end = timer()

        chosen_shard.time_cost = profile_timer_end - profile_timer_start - (end - start) # time for process running, assuming model is on device.
        #thread_lock.release()
        self.sleep_event.set()

    def lock_device(self, chosen):
        self.active_devices.append(chosen)
        self.available_devices.remove(chosen)
    
    def unlock_device(self, chosen):
        self.active_devices.remove(chosen)
        self.available_devices.append(chosen)
    
    def bar_gen(self, val_1, val_2):  
        percent_complete = val_1 / val_2
        bar_len = curses.COLS - curses.COLS / 10 - 2
        cols_full = int(percent_complete * bar_len)
        return "[{}{}]".format(int(cols_full) * '▓', int(bar_len-cols_full)*" ")
    
    def gui(self, stscr, dist_between, time, cache_task, cached_device_info, cache_success):
        stscr.clear()
        line_across = curses.COLS * "="
        mini_across = curses.COLS * "-"
        text = "HYDRA MODEL TRAINER v0.1"
        stscr.addstr (int(dist_between / 8), int(curses.COLS / 2) - int(len(text) / 2), text, curses.A_STANDOUT)
        
        
        device_string = ", "
        str_devices = [str(x) for x in self.all_devices]
        str_free = [str(round(get_used_space(x)/(1024*1024), 2))+"MB" for x in self.all_devices]
        
        if (cache_task is not None and cached_device_info is not None):
            text = " Task {} is cached on device {}. ".format(cache_task.name, cached_device_info)
            stscr.addstr(int(dist_between / 4), int(curses.COLS / 2) - int(len(text)/2), text)
        
        if (not cache_success):
            text = " Last Cache was a Failure. "
        else:
            text = " Last Cache was a Success. "
            
        stscr.addstr(int(3 * dist_between / 8), int(curses.COLS / 2) - int(len(text)/2), text)
        
        
        device_string = device_string.join(str_devices)
        free_string = "\n".join(str_free)
        text = " Devices {}. ".format(device_string)
        stscr.addstr(int(dist_between / 2), int(curses.COLS / 2) - int(len(text)/2), text)
        text = " Device Memory Consumption {} . ".format(str_free)
        stscr.addstr(5 * int(dist_between / 8), int(curses.COLS / 2) - int(len(text)/2), text)

        
        text = " {} seconds elapsed. ".format(round(time, 2))
        stscr.addstr(int(6 * dist_between / 8), int(curses.COLS / 2) - int(len(text)/2), text)
        
        stscr.addstr (int(7 * dist_between / 8), 0, line_across)

        
        for i in range(len(self.tasks)):
            y_pos = dist_between * (i+1)
            
            # task name
            stscr.addstr(y_pos, int(curses.COLS / 2) - int(len(text) / 2), text, curses.A_BOLD)
            
            completed_batches = (self.tasks[i].total_length - self.tasks[i].batches_remaining)
            
            completed_epochs = (self.tasks[i].total_epochs - self.tasks[i].epochs)
            
            # batch timer
            text = " {} out of {} shard-passes completed in this batch. ".format(self.tasks[i].curr_cycle, self.tasks[i].queue_len)
            stscr.addstr(y_pos + int(2 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            
            text = self.bar_gen(self.tasks[i].curr_cycle, self.tasks[i].queue_len)
            stscr.addstr(y_pos + int(3 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            
            # batch timer
            text = " {} out of {} batches completed in this epoch. ".format(completed_batches, self.tasks[i].total_length)
            stscr.addstr(y_pos + int(4 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            
            text = self.bar_gen(completed_batches, self.tasks[i].total_length)
            stscr.addstr(y_pos + int(5 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
                        
            # epoch timer
            text = " {} out of {} epochs completed.  ".format(completed_epochs, self.tasks[i].total_epochs)
            stscr.addstr(y_pos + int(7 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            
            text = self.bar_gen(completed_epochs, self.tasks[i].total_epochs)
            stscr.addstr(y_pos + int(8 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            
            max_time_remaining = (self.tasks[i].model.total_time + self.tasks[i].batch_time) * (self.tasks[i].batches_remaining + 1) * self.tasks[i].epochs
            # estimator
            text = "Estimated remaining time: {}s ".format(max_time_remaining)
            stscr.addstr(y_pos + int(9 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            text = "Elapsed active time: {}s. Last shard-pass took {}s".format(self.tasks[i].active_time, self.tasks[i].wastage)
            stscr.addstr(y_pos + int(11 * dist_between / 16), int(curses.COLS / 2) - int(len(text) / 2), text)
            
            
            stscr.addstr(y_pos+int(3 * dist_between / 4), 0, mini_across)


        
        stscr.addstr(curses.LINES - 2, 0, line_across)
        stscr.refresh()

    def train_models(self, gui=False):
        print("TRAINING STARTS")

        ctr = 0
        

        if gui:
            stscr = curses.initscr()
            curses.echo()
            window = curses.newwin(curses.LINES, curses.COLS, 0, 0)

            dist_between = int(curses.LINES / (len(self.tasks) + 1))

        old_time = 0

        cache_task = None # use this to try and "guess" the next task's completion time for that shard.
        cache_device = None
        
        CACHE_SYSTEM = True
        
        
        self.cached_tasks = [None for x in range(len(self.all_devices))]
        running_tasks = [None for x in range(len(self.all_devices))]
        
        global thread_lock
        
        
        # initial run
        
        
        
        for chosen_device in self.all_devices:
            task_times = [(i.model.total_time * i.batches_remaining) + i.total_length * i.model.total_time * i.epochs for i in self.idle_tasks]
            chosen_task = self.idle_tasks[np.argmax(task_times)]

            self.lock_device(chosen_device)
            chosen_task.setup_timing(chosen_device)
            self.active_tasks.append(chosen_task)
            running_tasks[chosen_device] = chosen_task
            chosen_shard = chosen_task.get_shard()
            
            self.idle_tasks.remove(chosen_task)
            print("Training task {}".format(chosen_task.name))
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
                        task_time = (i.model.total_time * i.batches_remaining ) + i.total_length * i.model.total_time * i.epochs
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
        while len(self.tasks) > 0:
            #ctr+=1
            #if (timer() - old_time > 0.5):        
            #    print  ([ task for task in self.active_tasks])
            try:
                self.sleep_event.wait()
                self.sleep_event.clear()
                if (len(self.tasks) == 0):
                    break
                ctr = 0
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
                            print("Training {} on {}".format(chosen_task.name, chosen_device))
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
                                task_time = (i.model.total_time * i.batches_remaining ) + i.total_length * i.model.total_time * i.epochs
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
                                task_time = (i.model.total_time * i.batches_remaining) + i.total_length * i.model.total_time * i.epochs
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
                print("TOTAL TIME TAKEN: {}".format(end - start))

                if (gui):
                    stscr.clear()
                    curses.nocbreak()
                    stscr.keypad(False)
                    curses.echo()
                    curses.endwin()
                sys.exit(0)

            
     
        end = timer()
        print("TOTAL TIME TAKEN: {}".format(end - start))
        if (gui):
            curses.nocbreak()
            stscr.keypad(False)
            curses.echo()
            curses.endwin()
