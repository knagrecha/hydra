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
from timeit import timeit as timer

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
    return loss_computer(out, labels)


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
                                collate_fn=lambda b: collate_batch(b, b_size, mask_frac, mask_id, cls_id), drop_last=True)




def get_model(name):
    layer_dictionary = {

        0: custom.BertEmbedding(28783, 1024, transpose=False),
        
        1: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        2: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        3: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        4: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 4
        
        5: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        6: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        7: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        8: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 8
        
        9: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        10: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        11: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        12: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 12
        
        13: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        14: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        15: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        16: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 16
        
        17: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        18: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        19: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        20: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 20
        
        21: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        22: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        23: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        24: custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 24 - BERT-Large Size (340M params)
        

        25: torch.nn.Linear(1024, 1024),
        26: torch.nn.GELU(),
        27: torch.nn.LayerNorm(1024, eps=1e-12),
        28: torch.nn.Linear(1024, 28783),
        29: pretraining_loss
    }
    
    io_dictionary = {
        29: [28, "label_0", "label_1"],
        28: [27],
        27: [26],
        26: [25],
        25: [24],
        24: [23],
        23: [22],
        22: [21],
        21: [20],
        20: [19],
        19: [18],
        18: [17],
        17: [16],
        16: [15],
        15: [14],
        14: [13],
        13: [12],
        12: [11],
        11: [10],
        10: [9],
        9: [8],
        8: [7],
        7: [6],
        6: [5],
        5: [4],
        4: [3],
        3: [2],
        2: [1],
        1: [0],
        0: ["batch_0"]
    }
    
    
    model_task = ModelTask(name, layer_dictionary, io_dictionary, get_data_loader(32), 0.001, 1)
    
    output_keys = ["batch_0"] + list(range(0, 30)) # the tensors that have forward receivers. 29's forward receiver is injected
                                                   # by the modelTask
    
    local_dictionary_0 = {idx: layer_dictionary[idx] for idx in range(0, 10)}
    local_input_dictionary_0 = {idx: io_dictionary[idx] for idx in range(0, 10)}
    #local_output_dictionary_0 = {output_keys[idx]: model_task.layer_to_output_dict[output_keys[idx]] for idx in range(0, 10)}
    #print(local_input_dictionary_0)


    local_dictionary_1 = {idx: layer_dictionary[idx] for idx in range(10, 20)}
    local_input_dictionary_1 = {idx: io_dictionary[idx] for idx in range(10, 20)}
    #local_output_dictionary_1 = {output_keys[idx]: model_task.layer_to_output_dict[output_keys[idx]] for idx in range(10, 20)}
    #print(local_output_dictionary_1)
    
    
    local_dictionary_2 = {idx: layer_dictionary[idx] for idx in range(20, 30)}
    local_input_dictionary_2 = {idx: io_dictionary[idx] for idx in range(20, 30)}
    #local_output_dictionary_2 = {output_keys[idx]: model_task.layer_to_output_dict[output_keys[idx]] for idx in range(20, 31)}

    
    
    
    
    
    shard_to_input_dict = {
        0: ["batch_0"],
        1: [9],
        2: [19, "label_0", "label_1"],
        3: [9],
        4: ["batch_0"]
    }
    
    shard_to_output_dict = {
        0: [9],
        1: [19],
        2: [29],
        3: [19],
        4: [9]
    }
    
    gen_0 = containers.GenericExecutor(local_dictionary_0, local_input_dictionary_0, shard_to_output_dict[0])
    sh_0_f = containers.ShardTask(gen_0, "f", 0.001)
    sh_0_b = containers.ShardTask(gen_0, "b", 0.001)
    
    gen_1 = containers.GenericExecutor(local_dictionary_1, local_input_dictionary_1, shard_to_output_dict[1])
    sh_1_f = containers.ShardTask(gen_1, "f", 0.001)
    sh_1_b = containers.ShardTask(gen_1, "b", 0.001)
    
    
    gen_2 = containers.GenericExecutor(local_dictionary_2, local_input_dictionary_2, shard_to_output_dict[2])
    sh_2_b = containers.ShardTask(gen_2, "b", 0.001)
        
    mini_batch_time = 2.5
    
    shard_dictionary = {
        0: sh_0_f,
        1: sh_1_f,
        2: sh_2_b,
        3: sh_1_b,
        4: sh_0_b
    }
    
    model_task.setup(None, shard_dictionary, shard_to_input_dict, shard_to_output_dict, mini_batch_time)
    return model_task


"""
    Main function
"""


def main():


    device_count = torch.cuda.device_count()

    model_0 = get_model("Model_0")
    model_1 = get_model("Model_1")
    model_2 = get_model("Model_2")
    
    with torch.no_grad():
        for model in [model_0, model_1, model_2]:
            total_loss = 0
            dataloader = get_data_loader(32, train=False)
            print("Evaluating: {}".format(model.name))

            out = None
            count = 0
            total_count = len(dataloader)
            for batch, label in dataloader:
                for key in range(0, 30):
                    if key == 0:
                        model.layer_dictionary[0] = model.layer_dictionary.to("cuda:0")
                        batch = batch.to("cuda:0")
                        out = model.layer_dictionary[0](batch)
                    elif key == 29:
                        label_0 = label[0].to("cuda:0")
                        label_1 = label[1].to("cuda:0")
                        loss = pretraining_loss(out, label_0, label_1)
                        total_loss += loss.item()
                    else:
                        model.layer_dictionary[key] = model.layer_dictionary.to("cuda:0")
                        batch = batch.to("cuda:0")
                        out = model.layer_dictionary[key] (out)
                count+=1
            print()
            print("Average batch loss: {}".format(total_loss / len(dataloader)))
                
    
    # create orchestrator
    orchestra = ModelOrchestrator([model_0, model_1, model_2])
    orchestra.verbose = 1

    orchestra.train_models()
                                                  
    with torch.no_grad():
        for model in [model_0, model_1, model_2]:
            total_loss = 0
            dataloader = get_data_loader(32, train=False)
            print("Evaluating: {}".format(model.name))
            count = 0
            total_count = len(dataloader)
            out = None
            for batch, label in dataloader:
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
    
if __name__ == "__main__":
    main()
