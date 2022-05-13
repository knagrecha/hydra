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

import hydra
from hydra import ModelTask, ModelOrchestrator
import customLayers as custom
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
import random
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
import os
import glob
from debugger import DebuggerGPT2LMHeadModel, GPT2EmbeddingLayer, GPT2OutputLayer
from datetime import datetime
import deepspeed
from utils import get_data_loader, get_data_set, collate_batch, ds_collate_batch, pretraining_loss

def checkpointed():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_layer = 35
    model = GPT2LMHeadModel(config=configuration) 
    torch.cuda.empty_cache()
    print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    b_size_p_device = 1
    training_args = TrainingArguments(gradient_checkpointing=True, output_dir="./output/", learning_rate=0.0001, per_device_train_batch_size=b_size_p_device, num_train_epochs=1)
    dataset = get_data_set()
    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=ds_collate_batch, optimizers=(torch.optim.SGD(model.parameters(), lr=0.0001), None))
    trainer.train()
        



def naive():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_layer = 25
    model = GPT2LMHeadModel(config=configuration) 
    torch.cuda.empty_cache()
    print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    b_size_p_device = 1
    training_args = TrainingArguments(gradient_checkpointing=False, output_dir="./output/", learning_rate=0.0001, per_device_train_batch_size=b_size_p_device,  num_train_epochs=1)
    dataset = get_data_set()
    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=ds_collate_batch, optimizers=(torch.optim.SGD(model.parameters(), lr=0.0001), None))
    trainer.train()
        


 
def deepspeed():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_layer = 65
    model = GPT2LMHeadModel(config=configuration) 
    torch.cuda.empty_cache()
    print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    b_size_p_device = 1
    training_args = TrainingArguments(gradient_checkpointing=False, output_dir="./output/", learning_rate=0.0001, per_device_train_batch_size=b_size_p_device, deepspeed="scaling_ds_cpu.json", num_train_epochs=1)
    dataset = get_data_set()
    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=ds_collate_batch, optimizers=(torch.optim.SGD(model.parameters(), lr=0.0001), None))
    trainer.train()
        
def deepspeed_train_with_checkpointing():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_layer = 115
    model = GPT2LMHeadModel(config=configuration) 
    torch.cuda.empty_cache()
    print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    b_size_p_device = 1
    training_args = TrainingArguments(gradient_checkpointing=True, output_dir="./output/", learning_rate=0.0001, per_device_train_batch_size=b_size_p_device, deepspeed="scaling_ds_cpu.json", num_train_epochs=1)
    dataset = get_data_set()
    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=ds_collate_batch, optimizers=(torch.optim.SGD(model.parameters(), lr=0.0001), None))
    trainer.train()
        
def hydra_train():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_layer = 100
    model = DebuggerGPT2LMHeadModel(config=configuration)
    print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    modules = [GPT2EmbeddingLayer(model.transformer.wte, model.transformer.wpe, model.transformer.drop)]
    for mod in model.transformer.h:
        modules.append(mod)
    modules.append(GPT2OutputLayer(model.transformer.ln_f))
    modules.append(model.lm_head)
    model = nn.Sequential(*modules) 
    task = ModelTask("test", model, pretraining_loss, get_data_loader(1), 0.0001, 1)
    orchestra = ModelOrchestrator([task])
    orchestra.verbose = 1
    orchestra.buffer = 35000
    orchestra.generate()
    orchestra.train_models()
        
        
            
def main():
    hydra_train()

if __name__ == "__main__":
    main()
