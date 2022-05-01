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
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import random
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
import os
import glob
from debugger import DebuggerGPT2LMHeadModel, GPT2EmbeddingLayer, GPT2OutputLayer
from datetime import datetime
import deepspeed
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #gpt2-medium
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})



def load_dataset(path=".data/wikitext-2/wiki.valid.tokens", combine=50000):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in paths:
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(tokenizer.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    print("RAW TEXT LENGTH: {}".format(len(raw_text)))       
    if raw_text:
        print("\n\n\n\ININININ\n\n\n\n")
        tokens = np.stack(tokenizer.encode(raw_text))
        print("TOKENS COUNT: {}".format(len(tokens)))
        token_chunks.append(tokens)
    return token_chunks


def collate_batch(batch):
    batch = torch.stack([torch.as_tensor(b) for b in batch], 0)
    return batch, batch.clone()


def get_data_loader(batch_size, context_length=1024):
    data = lazy_load()[0]
    print(len(data))

    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return data_loader

def lazy_load():
    cache_path = 'cache_path.npz'
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset(combine=1e99)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        #np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset(path=cache_path)
    assert len(data) > 0
    return data






def load_dataset_train(path=".data/wikitext-2/wiki.train.tokens", combine=50000):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in paths:
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(tokenizer.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    print("RAW TEXT LENGTH: {}".format(len(raw_text)))       
    if raw_text:
        print("\n\n\n\ININININ\n\n\n\n")
        tokens = np.stack(tokenizer.encode(raw_text))
        print("TOKENS COUNT: {}".format(len(tokens)))
        token_chunks.append(tokens)
    return token_chunks



def get_data_loader_train(batch_size, context_length=1024):
    data = lazy_load_train()[0]
    print(len(data))

    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return data_loader

def lazy_load_train():
    cache_path = 'cache_path_train_1.npz'
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset_train(combine=1e99)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        #np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset_train(path=cache_path)
    assert len(data) > 0
    return data




"""
    Custom loss function
"""

def pretraining_loss(lm_logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def get_base_model():
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    print(configuration)
    model = DebuggerGPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    params = sum(p.numel() for p in model.parameters())
    print("PARAMETER COUNT: {}".format(params))
    model.resize_token_embeddings(len(tokenizer))
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    return model

class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x):
        x = x.long()
        x = self.module(x)
        return x

def get_model_stack_for_checkpoint(model):
    
    modules = [ModuleWrapperIgnores2ndArg(GPT2EmbeddingLayer(model.transformer.wte, model.transformer.wpe, model.transformer.drop))]
    ctr = 0
    for mod in model.transformer.h:
        modules.append(mod)
        
            
    modules.append(GPT2OutputLayer(model.transformer.ln_f))
    modules.append(model.lm_head)
    new_model = nn.Sequential(*modules)
    print(new_model)
    return new_model


def get_model_stack(model):
    modules = [GPT2EmbeddingLayer(model.transformer.wte, model.transformer.wpe, model.transformer.drop)]
    for mod in model.transformer.h:
        modules.append(mod)
    modules.append(GPT2OutputLayer(model.transformer.ln_f))
    modules.append(model.lm_head)
    new_model = nn.Sequential(*modules)
    return new_model
    
def naive(base_model, dataloader):
    for i in range(10, 50):
        st = timer()
        torch.cuda.empty_cache()
        print("Testing stack of {} encoders".format(i))
        mod = get_model_stack(i, base_model)
        print("Parameters: {}".format(sum(p.numel() for p in mod.parameters())))
        sample, label = next(iter(dataloader))
        optimizer = torch.optim.SGD(mod.parameters(), lr=0.0001)
        mod = mod.to("cuda:0")
        sample = sample.to("cuda:0")
        label = label.to("cuda:0")
        out = mod(sample)
        loss = pretraining_loss(out, label)
        loss.backward()
        optimizer.step()
        mod.zero_grad()
        end = timer()
        del mod
        del loss
        del optimizer
        del out
        del sample
        del label
        print("TIME: {}".format(end-st))
        
        
class CheckPointedModel(nn.Module):
    def __init__(self, module, count):
        super().__init__()
        self.module = module
        self.count = count

    def forward(self,x):
        return torch.utils.checkpoint.checkpoint_sequential(self.module, self.count, x)
        
def checkpointed(base_model, dataloader):
    for i in range(45, 60):
        st = timer()
        torch.cuda.empty_cache()
        print("Testing stack of {} encoders".format(i))
        mod = get_model_stack_for_checkpoint(i, base_model)
        print("Parameters: {}".format(sum(p.numel() for p in mod.parameters())))
        sample, label = next(iter(dataloader))
        optimizer = torch.optim.SGD(mod.parameters(), lr=0.0001)
        mod = mod.to("cuda:0")
        sample = sample.to("cuda:0")
        sample = sample.type(torch.float32)
        sample.requires_grad_(True)
        label = label.to("cuda:0")
        out = torch.utils.checkpoint.checkpoint_sequential(mod, i, sample)
        loss = pretraining_loss(out, label)
        loss.backward()
        optimizer.step()
        mod.zero_grad()
        end = timer()
        del mod
        del loss
        del optimizer
        del out
        del sample
        del label
        print("TIME: {}".format(end-st))
        
        
def deepspeed_train(base_model, dataloader):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    print(args)
    
    st = timer()
    torch.cuda.empty_cache()
    mod = get_model_stack(base_model)
    print("\n\n\n\n\n\n\n\n\n")
    print("DeepSpeed Train Job")
    print(mod)
    model_engine, optimizer, _, _ = deepspeed.initialize(args, model=mod,optimizer = torch.optim.SGD(mod.parameters(), lr=0.0001))
    ctr = 0
    for sample, label in dataloader:
        print("BATCH: {} / {}".format(ctr, len(dataloader)), end='\r', flush=True)
        model_engine = model_engine.to("cuda:0")
        sample = sample.to("cuda:0")
        label = label.to("cuda:0")
        out = model_engine(sample)
        loss = pretraining_loss(out, label)
        model_engine.backward(loss)
        model_engine.step()
        del mod
        del model_engine
        del loss
        del optimizer
        del out
        del sample
        del label
    end = timer()
    print()
    print("TIME: {}".format(end-st))
        
def deepspeed_train_with_checkpointing(base_model, dataloader):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    print(args)
    for i in range(50, 75, 1):
        st = timer()
        torch.cuda.empty_cache()
        mod = CheckPointedModel(get_model_stack_for_checkpoint(i, base_model), i)
        print("\n\n\n\n\n\n\n\n\n")
        print("Testing stack of {} encoders".format(i))
        print("Parameters: {}".format(sum(p.numel() for p in mod.parameters())))
        print(mod)
        model_engine, optimizer, _, _ = deepspeed.initialize(args, model=mod,optimizer = torch.optim.SGD(mod.parameters(), lr=0.0001))
        
        sample, label = next(iter(dataloader))
        model_engine = model_engine.to("cuda:0")
        sample = sample.to("cuda:0")
        sample = sample.type(torch.float32)
        sample.requires_grad_(True)
        label = label.to("cuda:0")
        out = model_engine(sample)
        loss = pretraining_loss(out, label)
        model_engine.backward(loss)
        model_engine.step()
        end = timer()
        del mod
        del model_engine
        del loss
        del optimizer
        del out
        del sample
        del label
        print("TIME: {}".format(end-st))
        
            
def main():
    base_model = get_base_model()
    dataloader = get_data_loader_train(4)
    deepspeed_train(base_model, dataloader)

if __name__ == "__main__":
    main()
