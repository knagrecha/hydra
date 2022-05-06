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
from deepspeed.pipe import PipelineModule
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
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time, balance_by_size
from timeit import default_timer as timer
from utils import get_data_loader, get_data_loader_train, pretraining_loss, get_sequential_model, set_random_seed, collate_batch

            
def main(seed):
    set_random_seed(seed)
    
    lr_names = ["3e-4", "1e-4", "5e-5"]
    learning_rates = [3e-4, 1e-4, 5e-5]
    batch_sizes = [16, 12, 8]
    summed_runtimes = 0
    
    for idx, lr in enumerate(learning_rates):
        for b_size in batch_sizes:
            d_set = get_data_loader(b_size)
            new_model = get_sequential_model()
            sample, _ = next(iter(d_set))
            balance = [4, 7, 7, 7, 8, 7, 7, 4] # the automatic balancers OOM at b_size 16. This one works though
            new_model = GPipe(new_model, balance=balance, chunks=1) # remember that setting minibatch count to 1 is equivalent to naive model parallel. We let checkpointing stay
            optimizer = torch.optim.SGD(new_model.parameters(), lr = lr)
            total_len = len(d_set)
            ctr = 0
            st = timer()
            inner_timer = timer()
            for sample, label in d_set:
                ctr+=1
                time_taken = timer() - inner_timer
                inner_timer = timer()
                print("Time per iter: {}s | Estimated Total Runtime: {}".format(time_taken/ctr, time_taken/ctr * total_len), end='\r', flush=True) 
                sample = sample.to(new_model.devices[0])
                label = label.to(new_model.devices[-1])
                out = new_model(sample)
                loss = pretraining_loss(out, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del loss
                del out
                del label
                del sample

            time_taken = timer() - st
            print("Time Taken: {}".format(time_taken)) 
            summed_runtimes += time_taken
    print("TOTAL RUNTIME: {}".format(summed_runtimes))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    main(args.seed)
