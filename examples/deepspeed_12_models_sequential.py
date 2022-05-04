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
from deepspeed.profiling.flops_profiler import FlopsProfiler

from utils import get_data_set, get_data_set_train, pretraining_loss, get_base_model, set_random_seed, collate_batch
import deepspeed
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #gpt2-medium
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        sample, label = inputs
        outputs = model(sample)
        loss = pretraining_loss(outputs, label)
        return (loss, outputs) if return_outputs else loss

            
def main(seed):
    set_random_seed(seed)
    dataset = get_data_set()
    
    lr_names = ["3e-4", "1e-4", "5e-5"]
    learning_rates = [3e-4, 1e-4, 5e-5]
    learning_rates=[3e-4]
    batch_sizes = [1, 2, 4, 8]
    batch_sizes=[1]
    for idx, lr in enumerate(learning_rates):
        for b_size in batch_sizes:
            new_model = get_base_model()
            prof = FlopsProfiler(new_model)
            prof.start_profile()
            b_size_p_device = max(1, int(b_size / torch.cuda.device_count()))
            training_args = TrainingArguments(output_dir="./output/", learning_rate=lr, per_device_train_batch_size=b_size_p_device, deepspeed="scaling_ds_cpu.json", num_train_epochs=1)
            trainer = CustomTrainer(new_model, args=training_args, train_dataset=dataset, data_collator=collate_batch, optimizers=(torch.optim.SGD(new_model.parameters(), lr=lr), None))
            trainer.train()
            prof.print_model_profile()
            prof.end_profile()
        
    
    
    
    

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    main(args.seed)
