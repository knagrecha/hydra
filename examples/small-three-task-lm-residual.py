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
import hydra.components.partitioner.containers as containers
from hydra import ModelTask, ModelOrchestrator
import customLayers as custom
import copy
import torch
from torchtext.experimental.datasets import WikiText2
from torch.utils.data import DataLoader
from os import path
from timeit import default_timer as timer
import numpy as np


"""
    Preprocessing functions for the dataloaders.
"""



def process_raw_data(raw_data, batch_size, bptt):
    _num = raw_data.size(0) // (batch_size * bptt)
    raw_data = raw_data[:(_num * batch_size * bptt)]
    return raw_data

def collate_batch(batch_data, batch_size, mask_frac, mask_id, cls_id):
    #print(type(batch_data))
    batch_data = torch.tensor(batch_data)
    batch_data = batch_data.long()
    batch_data = batch_data.view(batch_size, -1)
    batch_data = batch_data.t().contiguous()
    # Generate masks with args.mask_frac
    data_len = batch_data.size(0)
    ones_num = int(data_len * mask_frac)
    zeros_num = data_len - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(data_len)]
    batch_data = torch.cat((torch.tensor([[cls_id] * batch_data.size(1)]).long(), batch_data))
    lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, [lm_mask, targets]

"""
    Custom loss function
"""

def pretraining_loss(out, lm_mask, labels):
    out = torch.stack([out[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
    loss_computer = torch.nn.CrossEntropyLoss()
    out = out.view(-1, 28783)
    x = loss_computer(out, labels)
    return x


"""
    Helper function to create a training dataloader.
"""
def get_data_loader(b_size, train=True):
    start = timer()
    print("\nPreparing to load dataset....")


    if (path.exists("vocab/torchtext_bert_vocab_wiki.pt")):
        vocab = torch.load("vocab/torchtext_bert_vocab_wiki.pt")
        if train:
            dataset = WikiText2(vocab=vocab, split='train') # set to train for real testing.pi
        else:
            dataset = WikiText2(vocab = vocab, split='test') # set to train for real testing.pi
    else:
        dataset = WikiText2(split='train') # set to train for real testing.pi
        vocab = dataset.get_vocab()
        torch.save(vocab, "vocab/torchtext_bert_vocab_wiki.pt")
        if train:
            dataset = WikiText2(vocab = vocab, split='train') # set to train for real testing.pi
        else:
            dataset = WikiText2(vocab = vocab, split='test') # set to train for real testing.pi


    dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, dataset)))

    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    bptt = 128
    mask_frac = 0.15
    end = timer()
    dataset = process_raw_data(dataset.data, b_size, bptt)
    print("Dataset Loaded in {} seconds.".format(end-start))



    return DataLoader(dataset, batch_size=b_size * bptt, shuffle=True,
                                collate_fn=lambda b: collate_batch(b, b_size, mask_frac, mask_id, cls_id), drop_last=True, pin_memory=True)




def get_model(name, layer_count):
    layer_dictionary = {}
    
    total_params = 0
    
    layer_dictionary[0] = custom.BertEmbedding(28783, 1024, transpose=False)
    for i in range(1, layer_count-5):
        layer_dictionary[i] = custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5)
    
    layer_dictionary[layer_count-5] = torch.nn.Linear(1024, 1024)
    layer_dictionary[layer_count-4] = torch.nn.GELU()
    layer_dictionary[layer_count-3] = torch.nn.LayerNorm(1024, eps=1e-12)
    layer_dictionary[layer_count-2] = torch.nn.Linear(1024, 28783)
    layer_dictionary[layer_count-1] = pretraining_loss
    
    print(len(layer_dictionary))

    io_dictionary = {}
    for i in range(len(layer_dictionary)):
        if i == 0:
            io_dictionary[i] = ["batch_0"]
        else:
            io_dictionary[i] = [i-1]
    io_dictionary[len(layer_dictionary) - 1].append("label_0")
    io_dictionary[len(layer_dictionary) - 1].append("label_1")
    
    
    model_task = ModelTask(name, layer_dictionary, io_dictionary, get_data_loader(32), 0.001, 1)
  
    curr_layer = 0
    excess = False
    
    f_shard_input = []
    b_shard_input = []
    f_shard_output = []
    b_shard_output = []
    f_shards = []
    b_shards = []
    while not excess:
        local_dictionary = {}
        local_input = {}
        for i in range(24):
            if i == 0:
                f_shard_input.append(io_dictionary[curr_layer])
                b_shard_input.append(io_dictionary[curr_layer])
            local_dictionary[curr_layer] = layer_dictionary[curr_layer]
            local_input[curr_layer] = io_dictionary[curr_layer]
            
            
            curr_layer+=1
            if curr_layer == len(layer_dictionary):
                excess = True
                f_shard_input.pop(-1)
                b_shard_input[-1] = b_shard_input[-1] + ["label_0", "label_1"]
                break
        
        if not excess:
            f_shard_output.append([curr_layer-1])
        b_shard_output.append([curr_layer-1])

        gen = containers.GenericExecutor(local_dictionary, local_input, b_shard_output[-1])
        total_params += sum(p.numel() for p in gen.parameters())
        
        if not excess:
            f_shards.append(containers.ShardTask(gen, "f", 0.001))
        
        b_shards.append(containers.ShardTask(gen, "b", 0.001))
    
    
    shard_to_input_dict = {}
    shard_to_output_dict = {}
    shard_dictionary = {}
    
    b_shard_input.reverse()
    b_shard_output.reverse()
    b_shards.reverse()
    all_input = f_shard_input + b_shard_input
    all_output = f_shard_output + b_shard_output
    all_shards = f_shards + b_shards

    for i in range(len(all_shards)):
        shard_to_input_dict[i] = all_input[i]
        shard_to_output_dict[i] = all_output[i]
        shard_dictionary[i] = all_shards[i]

    mini_batch_time = 2.5

    print("TOTAL PARAMETERS: {}".format(total_params))
    model_task.setup(None, shard_dictionary, shard_to_input_dict, shard_to_output_dict, mini_batch_time)
    return model_task


"""
    Main function
"""


def main():


    device_count = torch.cuda.device_count()

    model_0 = get_model("Model_0", 48)
    
    """
    print("RUNTIME COMPARISON")
    for model in [model_0]:
        total_loss = 0
        dataloader = get_data_loader(32)
        print("Evaluating: {}".format(model.name))
        count = 0
        total_count = len(dataloader)
        out = None
        params = set()
        for key, value in model.shard_dictionary.items():
            params.update( set(value.model.parameters() ) )
            
            
        optimizer = torch.optim.SGD(params, lr=0.001)
        
        full_st = timer()
        for batch, label in dataloader:
            if (count % 1 == 0):
                print(" {} / {}".format(count, total_count))
                f_times_sum = 0
                b_times_sum = 0
            for key in range(0, len(model.layer_dictionary)):
                if key == 0:
                    model.layer_dictionary[0] = model.layer_dictionary[0].to("cuda:0")
                    batch = batch.to("cuda:0")
                    out = model.layer_dictionary[0](batch)
                elif key == len(model.layer_dictionary) - 1:
                    label_0 = label[0].to("cuda:0")
                    label_1 = label[1].to("cuda:0")
                    loss = pretraining_loss(out, label_0, label_1)
                    st = timer()
                    loss.backward()
                    optimizer.step()
                    for key, shard in model.shard_dictionary.items():
                        shard.model.zero_grad()
                    total_loss += loss.item()
                    end = timer()
                else:
                    model.layer_dictionary[key] = model.layer_dictionary[key].to("cuda:0")
                    out = out.to("cuda:0")
                    out = model.layer_dictionary[key] (out)
            count+=1
        print("TIME TAKEN: {}".format(timer() - full_st))
        print("Average batch loss: {}".format(total_loss / len(dataloader)))
    
    """
    """
    with torch.no_grad():
        for model in [model_0, model_1]:
            total_loss = 0
            dataloader = get_data_loader(32, train=False)
            print("Evaluating: {}".format(model.name))
            out = None
            count = 0
            total_count = len(dataloader)
            for batch, label in dataloader:
                if (count == 20):
                    st = timer()

                if (count % 10 == 0):
                    print(" {} / {}".format(count, total_count))
                for key in range(0, 30):
                    if key == 0:
                        model.layer_dictionary[0] = model.layer_dictionary[0].to("cuda:0")
                        batch = batch.to("cuda:0")
                        out = model.layer_dictionary[0](batch)
                    elif key == 29:
                        label_0 = label[0].to("cuda:0")
                        label_1 = label[1].to("cuda:0")
                        loss = pretraining_loss(out, label_0, label_1)
                        total_loss += loss.item()
                    else:
                        model.layer_dictionary[key] = model.layer_dictionary[key].to("cuda:0")
                        batch = batch.to("cuda:0")
                        out = model.layer_dictionary[key] (out)
                        
                if (count == 20):

                    end = timer()
                    print("MiniBatch Time: {}".format(end - st))
                    
                count+=1
                
            print()
            print("Average batch loss: {}".format(total_loss / len(dataloader)))
    """
        
   
    # create orchestrator
    orchestra = ModelOrchestrator([model_0])
    orchestra.verbose = 1

    st = timer()
    orchestra.train_models()
    print("TIME TAKEN: {}".format(timer() - st))

    """
    with torch.no_grad():
        for model in [model_0, model_1]:
            total_loss = 0
            dataloader = get_data_loader(32, train=False)
            print("Evaluating: {}".format(model.name))
            count = 0
            total_count = len(dataloader)
            out = None
            for batch, label in dataloader:
                if (count % 10 == 0):
                    print(" {} / {}".format(count, total_count))
                for key in range(0, 30):
                    if key == 0:
                        model.layer_dictionary[0] = model.layer_dictionary[0].to("cuda:0")
                        batch = batch.to("cuda:0")
                        out = model.layer_dictionary[0](batch)
                    elif key == 29:
                        label_0 = label[0].to("cuda:0")
                        label_1 = label[1].to("cuda:0")
                        loss = pretraining_loss(out, label_0, label_1)
                        total_loss += loss.item()
                    else:
                        model.layer_dictionary[key] = model.layer_dictionary[key].to("cuda:0")
                        batch = batch.to("cuda:0")
                        out = model.layer_dictionary[key] (out)
                count+=1
            print()
            print("Average batch loss: {}".format(total_loss / len(dataloader)))
    """


if __name__ == "__main__":
    main()
