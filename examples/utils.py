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
import torch
import torch.nn as nn
import argparse
from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader
from os import path
from timeit import timeit as timer
import gc
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from debugger import DebuggerGPT2LMHeadModel, GPT2EmbeddingLayer, GPT2OutputLayer
import random
import numpy as np
from torch.nn import CrossEntropyLoss
import math
import os
import glob
from datetime import datetime


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl') #gpt2-medium
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    


def load_dataset(path=".data/wikitext-2/wiki.test.tokens", combine=50000):
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
    if raw_text:
        tokens = np.stack(tokenizer.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def collate_batch(batch):
    batch = torch.stack([torch.as_tensor(b) for b in batch], 0)
    return batch, batch.clone()


def ds_collate_batch(batch):
    batch = torch.stack([torch.as_tensor(b) for b in batch], 0)
    return {"input_ids": batch, "labels": batch.clone()}


def get_data_set( context_length=512):
    data = lazy_load()[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    return ds


def get_data_loader(batch_size, context_length=512):
    data = lazy_load()[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return data_loader

def lazy_load():
    cache_path = 'cache_path_test.npz'
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset(combine=1e99)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)
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
    if raw_text:
        tokens = np.stack(tokenizer.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def get_data_loader_hydra(batch_size, context_length=512):
    data = lazy_load_train()[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, pin_memory=True)

    return data_loader



def get_data_loader_train(batch_size, context_length=512):
    data = lazy_load_train()[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return data_loader


def get_data_set_train(context_length=512):
    data = lazy_load_train()[0]
    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+context_length) 
        for i in range(0, len(data) - (len(data) % context_length), context_length)])
    return ds

def lazy_load_train():
    cache_path = 'cache_path_train.npz'
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset_train(combine=1e99)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset_train(path=cache_path)
    assert len(data) > 0
    return data


def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


"""
    Custom loss function
"""

def pretraining_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.view(-1)
    
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def get_base_model():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_ctx = 512
    configuration.gradient_checkpointing = True
    configuration.use_cache = False
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=configuration)
    params = sum(p.numel() for p in model.parameters())
    model.resize_token_embeddings(len(tokenizer))
    return model

class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x):
        x = x.long()
        x = self.module(x)
        return x


def get_ckpt_model():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_ctx = 512
    model = DebuggerGPT2LMHeadModel.from_pretrained("gpt2-xl", config=configuration)
    modules = [ModuleWrapperIgnores2ndArg(GPT2EmbeddingLayer(model.transformer.wte, model.transformer.wpe, model.transformer.drop))]
    for mod in model.transformer.h:
        modules.append(mod)
    modules.append(GPT2OutputLayer(model.transformer.ln_f))
    modules.append(model.lm_head)
    params = sum(p.numel() for p in model.parameters())
    model.resize_token_embeddings(len(tokenizer))
    return nn.Sequential(*modules)


def get_sequential_model():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    configuration.n_ctx = 512
    model = DebuggerGPT2LMHeadModel.from_pretrained("gpt2-xl", config=configuration)
    modules = [GPT2EmbeddingLayer(model.transformer.wte, model.transformer.wpe, model.transformer.drop)]
    for mod in model.transformer.h:
        modules.append(mod)
    modules.append(GPT2OutputLayer(model.transformer.ln_f))
    modules.append(model.lm_head)
    params = sum(p.numel() for p in model.parameters())
    model.resize_token_embeddings(len(tokenizer))
    return nn.Sequential(*modules)
