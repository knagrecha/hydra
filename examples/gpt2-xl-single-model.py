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



def get_model():
    configuration = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=False)
    print(configuration)
    model = DebuggerGPT2LMHeadModel.from_pretrained("gpt2-xl", config=configuration)
    params = sum(p.numel() for p in model.parameters())
    print("PARAMETER COUNT: {}".format(params))
    model.resize_token_embeddings(len(tokenizer))

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    modules = [GPT2EmbeddingLayer(model.transformer.wte, model.transformer.wpe, model.transformer.drop)]
    for mod in model.transformer.h:
        modules.append(mod)
    modules.append(GPT2OutputLayer(model.transformer.ln_f))
    modules.append(model.lm_head)
    model = nn.Sequential(*modules)
    
    return model
    

def main():
    device_count = torch.cuda.device_count()

    model_0 = get_model()
    
  
    params = sum(p.numel() for p in model_0.parameters())
    print("Total parameters: {}".format(params))
    
    dataloader_0 = get_data_loader_train(8) # Generate dataloader
    sample, label = next(iter(dataloader_0))
    print(sample.shape)
    #dataloader_1 = get_data_loader_train(32)
    #dataloader_2 = get_data_loader_train(32)
    
    

    
    task_0 = ModelTask("Model 0", model_0, pretraining_loss, dataloader_0, 0.001, 1)
    #task_1 = ModelTask("Model 1", model_1, pretraining_loss, dataloader_1, 0.001, 4)
    #task_2 = ModelTask("Model 2", model_2, pretraining_loss, dataloader_2, 0.001, 4)
    
    


    # create orchestrator
    orchestra = ModelOrchestrator([task_0])
    orchestra.verbose = 1
    
    """
     The Double-Buffer. Adjusting this up or down a bit can help to address minor
     errors in partitioning memory consumption.
    """
    
    orchestra.buffer = 5000

    orchestra.generate()
    orchestra.train_models()
    
    print("==========SEQUENTIALIZED MODEL!!!========")
    valid_loader = get_data_loader(2)
    ctr = 0
    with torch.no_grad():
        accum_loss = 0
        for sample, label in valid_loader:
            print("SAMPLE: {} / {}".format(ctr, len(valid_loader)))
            ctr+=1
            outputs = model_0(sample)
            my_loss = pretraining_loss(outputs, label)
            print("PPL: {}".format(math.exp(my_loss)))
            accum_loss += my_loss.item()
            print("RUNNING PPL: {}".format(math.exp(accum_loss/ctr)))
    print("ZERO SHOT TRAINING LOSS: {}".format(math.exp(accum_loss/ctr)))
    
    
    torch.save(model_0, "model_{}-{}-{}".format(datetime.now().hour, datetime.now().minute, datetime.now().second))
if __name__ == "__main__":
    main()
