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


# Scaled down for demo purposes. Recommend training with 2-4 GPUs.

import copy
import torch
import torch.nn as nn
import argparse
from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader
from os import path
from timeit import timeit as timer
import gc
from hydra.utilities import get_free_space
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler, Subset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from debugger import DebuggerGPT2LMHeadModel
import random
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
import os
import glob
from datetime import datetime
from timeit import default_timer as timer
from utils import get_data_loader, get_data_loader_train, pretraining_loss, get_ckpt_model, set_random_seed, collate_batch

            
def main(seed):
    set_random_seed(seed)
    
    lr_names = ["3e-4", "1e-4", "5e-5"]
    
    learning_rates = [3e-4, 1e-4, 5e-5, 6e-5, 1e-5, 2e-5]
    batch_sizes = [8, 16]
    summed_runtimes = 0
    
    for idx, lr in enumerate(learning_rates):
        for b_size in batch_sizes:
            d_set = get_data_loader_train(b_size)
            new_model = get_ckpt_model()
            optimizer = torch.optim.SGD(new_model.parameters(), lr = lr)
            modules = []
            boundaries = [13, 13, 13, 12]
            children = list(new_model.children())
            total_ctr = 0
            assert sum(boundaries) == len(children)
            
            current_module = []
            ctr = 0
            while (ctr < boundaries[0]):
                current_module.append(children[total_ctr])
                ctr+=1
                total_ctr+=1
            modules.append(nn.Sequential(*current_module).to("cuda:0"))

            current_module = []
            ctr = 0
            while (ctr < boundaries[1]):
                current_module.append(children[total_ctr])
                ctr+=1
                total_ctr+=1
            modules.append(nn.Sequential(*current_module).to("cuda:1"))


            current_module = []
            ctr = 0
            while (ctr < boundaries[2]):
                current_module.append(children[total_ctr])
                ctr+=1
                total_ctr+=1
            modules.append(nn.Sequential(*current_module).to("cuda:2"))


            current_module = []
            ctr = 0
            while (ctr < boundaries[3]):
                current_module.append(children[total_ctr])
                ctr+=1
                total_ctr+=1
            modules.append(nn.Sequential(*current_module).to("cuda:3"))

            total_len = len(d_set)
            ctr = 0
            st = timer()
            end = timer()
            for sample, label in d_set:
                ctr+=1
                optimizer_0 = torch.optim.SGD(modules[0].parameters(), lr = lr)
                optimizer_1 = torch.optim.SGD(modules[1].parameters(), lr = lr)
                optimizer_2 = torch.optim.SGD(modules[2].parameters(), lr = lr)
                optimizer_3 = torch.optim.SGD(modules[3].parameters(), lr = lr)
                print("{} / {} | Minibatch time: {}".format(ctr, total_len, timer()-end), end='\r', flush=True) 
                end = timer()
                sample = sample.to("cuda:0")
                sample = sample.type(torch.float32)
                sample.requires_grad_(True)
                label = label.to("cuda:3")
                sample_0 = torch.utils.checkpoint.checkpoint_sequential(modules[0], boundaries[0], sample)
                sample_0 = sample_0.to("cuda:1")
                sample_1 = torch.utils.checkpoint.checkpoint_sequential(modules[1], boundaries[1], sample_0)
                sample_1 = sample_1.to("cuda:2")
                sample_2 = torch.utils.checkpoint.checkpoint_sequential(modules[2], boundaries[2], sample_1)
                sample_2 = sample_2.to("cuda:3")
                sample_3 = torch.utils.checkpoint.checkpoint_sequential(modules[3], boundaries[3], sample_2)
                loss = pretraining_loss(sample_3, label)
                loss.backward()
                optimizer_0.step()
                optimizer_0.zero_grad()
                optimizer_1.step()
                optimizer_1.zero_grad()
                optimizer_2.step()
                optimizer_2.zero_grad()

                optimizer_3.step()
                optimizer_3.zero_grad()


                for model in modules:
                    model.zero_grad()
                del optimizer_0
                del optimizer_1
                del optimizer_2
                del optimizer_3
                del loss
                del label
                del sample
                del sample_0
                del sample_1
                del sample_2
                del sample_3
                gc.collect()
                torch.cuda.empty_cache()

            time_taken = timer() - st
            print("Time Taken: {}".format(time_taken)) 
            summed_runtimes += time_taken
    print("TOTAL RUNTIME: {}".format(summed_runtimes))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    main(args.seed)
