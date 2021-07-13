
import torch.nn as nn
from .utilities import get_free_space
import numpy as np
import gc
from timeit import default_timer as timer

import torch
from .Container import NNContainer
import traceback



class ShardedTask():

    def __init__(self, model, direction, time_taken, idx):
        
        self.model = model
        self.direction = direction
        self.time_cost = time_taken
        self.idx = idx
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01)



class Model():
    def __init__(self, model):
        self.f_shards = []
        self.b_shards = []
        # this info is nice to have for the scheduler
        self.shard_forward_times = []
        self.shard_backward_times = []
        self.shard_times = []
        self.total_time = 0

        self.verbose = 0

        self.layers = list(model.children()) # ideally we'd use model.modules. But that could lead to
                                             # issues in our sharding algo. TODO

    def setup_manual(self, true_criterion, batch_orig, indices):
        shard_idx = 0
        true_labels = batch_orig[-1]
        batch_orig = batch_orig[0:len(batch_orig)-1]
        device = torch.device("cuda:0")
        batch = [x.to(device) for x in batch_orig]
        
        for shard_starts in range(len(indices) - 1):
            list_of_layers = self.layers[indices[shard_starts]:indices[shard_starts+1]]
            print("Layers {} to {}".format(indices[shard_starts], indices[shard_starts+1]))
            start_f = timer()
            model = NNContainer(list_of_layers)
            if (isinstance(batch, list) or isinstance(batch, tuple)):
                batch = [x.to(device) for x in batch]
            else:
                batch = batch.to(device)
            model.to(device)
            out = model(batch)
            end_f = timer()
            start_b = timer()
            labels = out.detach().clone()
            criterion = nn.MSELoss()
            loss = criterion(out, labels)
            print("loss")
            if loss.requires_grad:
                loss.backward()
                model.zero_grad()
            model.zero_grad()
            del loss
            del criterion
            del labels
            
            if (isinstance(batch, list) or isinstance(batch, tuple)):
                batch = [x.cpu() for x in batch]
            del batch
            batch = out.cpu().detach().clone()
            del out
            

            model.to(torch.device("cpu"))  # this is an inplace operation
            end_b = timer()
            
            
            self.f_shards.append(ShardedTask(model, "f", end_f - start_f, shard_idx))
            self.b_shards.append(ShardedTask(model, "b", end_b - start_b, shard_idx))
            self.total_time = self.total_time + (end_f - start_f) + (end_b - start_b)
            shard_idx+=1
            
        print("Layers {} to {}".format(indices[-1], len(self.layers)-1))
        list_of_layers = self.layers[indices[-1]:]
        start_f = timer()
        model = NNContainer(list_of_layers)
        if (isinstance(batch, list) or isinstance(batch, tuple)):
            batch = [x.to(device) for x in batch]
        else:
            batch = batch.to(device)
        model.to(device)
        out = model(batch)
        end_f = timer()
        start_b = timer()
        
        criterion = nn.MSELoss()
        
        if not isinstance(true_labels, torch.Tensor):
            true_labels = [x.to(device, non_blocking=True) for x in true_labels]
        else:
            true_labels = true_labels.to(device, non_blocking=True)
        loss = true_criterion(out, true_labels)
        loss.backward()
        model.zero_grad()
        del loss
        del criterion
        del true_labels
        del out
        if (isinstance(batch, list) or isinstance(batch, tuple)):
            batch = [x.cpu() for x in batch]
        del batch


        model.to(torch.device("cpu"))  # this is an inplace operation
        end_b = timer()


        self.f_shards.append(ShardedTask(model, "f", end_f - start_f, shard_idx))
        self.b_shards.append(ShardedTask(model, "b", end_b - start_b, shard_idx))
        self.total_time = self.total_time + (end_f - start_f) + (end_b - start_b)

        
        self.b_shards.reverse()
        self.b_shards.pop(0)
        
    
    
    def setup(self, true_criterion, batch_orig, buffer):
        
        available_gpus = torch.cuda.device_count()
        available_devices = list(range(available_gpus))
        free_spaces = [get_free_space(x) for x in available_devices]
        
        device_idx = np.argmin(free_spaces)
        device = torch.device("cuda:"+str(device_idx))
        if (self.verbose == 1):
            print(free_spaces)
            print("Experimental sharding will occur on device {}.".format(device_idx))
    
        if (buffer != None):
            buffer_arr = torch.zeros((buffer, buffer)).to(device)
        else:
            buffer_arr = torch.zeros((15000, 15000)).to(device)
            print("Buffer Arr created.")
            print([get_free_space(x) for x in available_devices])
            
        true_labels = batch_orig[-1]
        batch_orig = batch_orig[0:len(batch_orig)-1]
        
        batch = [x.to(device) for x in batch_orig]
        
        print("Batch Transferred.")
        print([get_free_space(x) for x in available_devices])

        
        list_of_layers = []

        true_layer_index = 0
        shard_idx = 0
        #print("Free memory: " + str(get_free_space()))
        sequence_grads = []
        sequence_outs = []
        
        while true_layer_index < (len(self.layers)):
            if not (isinstance(batch, torch.Tensor)):
                batch = [x.to(device) for x in batch_orig]
            else:
                batch = batch.to(device)
            
            
            oom = False
            oom_override = False
            #if (self.verbose == 1):
            #    print("Current Memory Free {0} MB\t".format(get_free_space(device_idx)/(1024*1024)))
            
            
            try:
                list_of_layers.append(self.layers[true_layer_index])
                self.layers[true_layer_index] = self.layers[true_layer_index].to(device)
                
                if (isinstance(batch, tuple) or isinstance(batch, list)):
                    #print (mod)
                    out = self.layers[true_layer_index](*batch)
                else:
                    out = self.layers[true_layer_index](batch)
                
                
                if not (isinstance(out, torch.Tensor)):
                    grads = []
                    new_out = []
                    for output in out:
                        if output.requires_grad:
                            grads.append(torch.ones_like(output).to(device))
                            new_out.append(output)
                    if (len(new_out)!= 0):
                        torch.autograd.backward(new_out, grads, retain_graph = True)
                        self.layers[true_layer_index].zero_grad()

                else:
                    if (out.requires_grad):
                        grads = []
                        grads.append(torch.ones_like(out).to(device))
                        torch.autograd.backward(out, grads, retain_graph = True)
                        self.layers[true_layer_index].zero_grad()

                sequence_outs.append(out)
                #sequence_grads.append(grads)
                #print("Pass run.")
                print("Layer Index: {} | Memory {}".format(true_layer_index, [get_free_space(x) for x in available_devices]))
                
                # GPU Memory Consumption is complete
                # if we reach this point we can safely assume the pass is safe. (i.e. batch will soon be replaced by out - why hold onto it?)
                # so if we DO NOT reach this point, batch is unmodified (i.e. ready for a new pass in a new subset)
                
                del batch
                if not (isinstance(out, torch.Tensor)):
                    batch = [x.cpu().detach().clone() for x in out]
                batch = out.cpu().detach().clone()
                true_layer_index+=1
                
             
            except Exception as e:
                print(e)
                traceback.print_exc()
                oom = True
                
            if (oom):
                true_layer_index -=1
                print("Split at layer {}".format(true_layer_index))
                pioneer_layer = list_of_layers.pop()
                pioneer_layer = pioneer_layer.cpu()
                if (len(list_of_layers) == 0):
                    raise RuntimeError("Your minimum defined module's size is too large! Try chunking your modules into smaller pieces?")
                
                
                del sequence_outs
                sequence_outs = []
                
                oom = False
                if not (isinstance(batch_orig, torch.Tensor)):
                    batch_orig = [x.to(device) for x in batch_orig]
                else:
                    batch_orig = batch_orig.to(device)
                start_f = timer() # used for scheduler

                model = NNContainer(nn.ModuleList(list_of_layers))
                model.to(device)

                    
                out = model(batch_orig)
                end_f = timer()

                start_b = timer() # used for scheduler
                
                labels = out.detach().clone()
                criterion = nn.MSELoss()
                loss = criterion(out, labels)
                loss.backward()
                model.zero_grad()
                if not (isinstance(out, torch.Tensor)):
                    batch_orig = [x.cpu().detach().clone() for x in out]
                else:
                    batch_orig = out.cpu().detach().clone()
                    
                del out
                
                del loss
                del criterion
                del labels

                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                model.cpu()  # this is an inplace operation


                end_b = timer()

                #model.share_memory()
                self.f_shards.append(ShardedTask(model, "f", end_f-start_f, shard_idx))
                self.b_shards.append(ShardedTask(model, "b", end_b-start_b, shard_idx))
                shard_idx+=1

                self.total_time = self.total_time + (end_f - start_f) + (end_b - start_b)
                


                list_of_layers = []
                list_of_layers.append(pioneer_layer)
                print("Resuming at {} | Memory {}".format(pioneer_layer, [get_free_space(x) for x in available_devices]))
        
                #print([get_free_space(x) for x in available_devices])

        if (len(list_of_layers) > 0):
            del sequence_outs
            sequence_outs = []
            if not (isinstance(batch_orig, torch.Tensor)):
                batch_orig = [x.to(device) for x in batch_orig]
            else:
                batch_orig = batch_orig.to(device)
            
            start_f = timer() # used for scheduler
            
            model = NNContainer(nn.ModuleList(list_of_layers))
            model.to(device)
            
            out = model(batch_orig)
            
            end_f = timer()
            self.shard_forward_times.append(end_f-start_f)

            
            start_b = timer() # used for scheduler
            if not isinstance(true_labels, torch.Tensor):
                true_labels = [x.to(device, non_blocking=True) for x in true_labels]
            else:
                true_labels = true_labels.to(device, non_blocking=True)

            loss = true_criterion(out, true_labels)
            loss.backward()
            model.zero_grad()
            del loss
            del true_criterion
            del true_labels
            del out
            if (isinstance(batch, list) or isinstance(batch, tuple)):
                batch = [x.cpu() for x in batch]
            del batch

            model.to(torch.device("cpu"))  # this is an inplace operation

            end_b = timer()
            self.f_shards.append(ShardedTask(model, "f", end_f - start_f, shard_idx))
            self.b_shards.append(ShardedTask(model, "b", end_b - start_b, shard_idx))

            self.total_time = self.total_time + (end_f - start_f) + (end_b - start_b)
            print("Expected model minibatch time: {}".format(self.total_time))

            self.b_shards.reverse()
            self.b_shards.pop(0)
            print("Resuming at Memory {}".format([get_free_space(x) for x in available_devices]))
        if (buffer_arr != None):
            print("Clearing out the buffer array.")
            del buffer_arr
            torch.cuda.empty_cache()
            print("Resuming at emory {}".format([get_free_space(x) for x in available_devices]))


