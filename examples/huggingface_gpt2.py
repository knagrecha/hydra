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
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import random
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math


tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #gpt2-medium
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class GPT2Dataset(Dataset):

    def __init__(self, base_dataset, gpt2_type="gpt2", max_length=768, data_length=0):

        self.tokenizer = tokenizer

        self.input_ids = []
        self.attn_masks = []

        for idx, batch in enumerate(base_dataset):
            print("TOKENIZING: {} / {}".format(idx, data_length), end='\r', flush=True)
            encodings_dict = self.tokenizer('<|startoftext|>'+ batch + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 



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


"""
    Helper function to create a training dataloader.
"""

def get_data_loaders(b_size):
    start = timer()
    print("\nPreparing to load dataset....")
    tr_dataset = WikiText2(split='train') # set to train for real testing
    tr_dataset = GPT2Dataset(tr_dataset, data_length=36718) # create the tokenized dataset

    valid_dataset = WikiText2(split='valid') # set to train for real testing
    valid_dataset = GPT2Dataset(valid_dataset, data_length=3760) # create the tokenized dataset


    train_dataloader = DataLoader(
            tr_dataset,  # The training samples.
            sampler = RandomSampler(tr_dataset), # Select batches randomly
            batch_size = b_size # Trains with this batch size.
        )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                valid_dataset, # The validation samples.
                sampler = SequentialSampler(valid_dataset), # Pull out batches sequentially.
                batch_size = b_size # Evaluate with this batch size.
            )


    print()
    print("DATASET PREPPED")
    return train_dataloader, validation_dataloader


def get_model():
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    params = sum(p.numel() for p in model.parameters())
    print("PARAMETER COUNT: {}".format(params))
    model.resize_token_embeddings(len(tokenizer))

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    return model
    

def main():
    device_count = torch.cuda.device_count()

    model_0 = get_model()
    print(model_0)
    train_loader, valid_loader = get_data_loaders(1)
    ctr = 0
    with torch.no_grad():
        accum_loss = 0
        for sample in valid_loader:
            print("SAMPLE: {} / {}".format(ctr, len(valid_loader)))
            b_input_ids = sample[0]
            b_labels = sample[0]
            b_masks = sample[1]
            ctr+=1
            outputs = model_0(b_input_ids, attention_mask=b_masks, labels=b_labels)
            my_loss = pretraining_loss(outputs[1], b_labels)
            print("PPL: {}".format(math.exp(my_loss)))
            accum_loss += my_loss.item()
            print("RUNNING PPL: {}".format(math.exp(accum_loss/ctr)))
    print("ZERO SHOT TRAINING LOSS: {}".format(math.exp(accum_loss/ctr)))
    
    
    #model_1 = get_model()
    #model_2 = get_model()
    
    """
    params = sum(p.numel() for p in model_0.parameters())
    print("Total parameters: {}".format(params))
    
    dataloader_0 = get_data_loader_train(128) # Generate dataloader
    #dataloader_1 = get_data_loader_train(32)
    #dataloader_2 = get_data_loader_train(32)
    
    

    
    task_0 = ModelTask("Model 0", model_0, pretraining_loss, dataloader_0, 0.001, 1)
    #task_1 = ModelTask("Model 1", model_1, pretraining_loss, dataloader_1, 0.001, 4)
    #task_2 = ModelTask("Model 2", model_2, pretraining_loss, dataloader_2, 0.001, 4)
    
    


    # create orchestrator
    orchestra = ModelOrchestrator([task_0])
    orchestra.verbose = 1
    """
    """
     The Double-Buffer. Adjusting this up or down a bit can help to address minor
     errors in partitioning memory consumption.
    """
    """
    orchestra.buffer = 10000

    orchestra.generate()
    orchestra.train_models()
    """
if __name__ == "__main__":
    main()
