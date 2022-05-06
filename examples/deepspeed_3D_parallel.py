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
from deepspeed.pipe import PipelineModule

from utils import get_data_set, get_data_set_train, pretraining_loss, get_sequential_model, set_random_seed, collate_batch
import deepspeed
            
def main(seed):
    set_random_seed(seed)
    
    deepspeed.init_distributed()
    lr_names = ["3e-4", "1e-4", "5e-5", "6e-5"]
    learning_rates = [3e-4, 1e-4, 5e-5, 6e-5]
    batch_sizes = [16, 12, 8]
    for idx, lr in enumerate(learning_rates):
        for b_size in batch_sizes:
            d_set = get_data_set_train()
            new_model = get_sequential_model()
            new_model = PipelineModule(layers=new_model, num_stages=torch.cuda.device_count(), loss_fn=pretraining_loss, activation_checkpoint_interval = 1, base_seed = seed)
            engine, _, _, _ = deepspeed.initialize(
                args=args,
                model=new_model,
                model_parameters=[p for p in new_model.parameters() if p.requires_grad],
                training_data=d_set, collate_fn=collate_batch)
            
            prof = FlopsProfiler(new_model)
            prof.start_profile()
            for step in range(10):
                engine.train_batch()
            prof.print_model_profile()
            prof.end_profile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    main(args.seed)
