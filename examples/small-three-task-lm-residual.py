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

def pretraining_loss(out, targets):
    lm_mask, label = targets
    out = torch.stack([out[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
    loss_computer = torch.nn.CrossEntropyLoss()
    out = out.view(-1, 28783)
    label = label.to(out.device)

    return loss_computer(out, label)


"""
    Helper function to create a training dataloader.
"""

def get_data_loader_train(b_size):
    start = timer()
    print("\nPreparing to load dataset....")


    if (path.exists("vocab/torchtext_bert_vocab_wiki.pt")):
        vocab = torch.load("vocab/torchtext_bert_vocab_wiki.pt")
        dataset = WikiText2(vocab=vocab, split='train') # set to train for real testing.pi
    else:
        dataset = WikiText2(split='train') # set to train for real testing.pi
        vocab = dataset.get_vocab()
        torch.save(vocab, "vocab/torchtext_bert_vocab_wiki.pt")
        dataset = WikiText2(vocab = vocab, split='train') # set to train for real testing.pi


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




"""
    Main function
"""


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
        29: [28, "label_0"],
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
    
    local_dictionary_0 = {idx: layer_dictionary[idx] for idx in range(0, 10)}
    local_input_dictionary_0 = {idx: io_dictionary[idx] for idx in range(0, 10)}

    gen_0 = containers.GenericExecutor(local_dictionary_0, local_input_dictionary_0)
    sh_0_f = containers.ShardTask(gen_0, "f", 0.001)
    sh_0_b = containers.ShardTask(gen_0, "b", 0.001)
    
    local_dictionary_1 = {idx: layer_dictionary[idx] for idx in range(10, 20)}
    local_input_dictionary_1 = {idx: io_dictionary[idx] for idx in range(10, 20)}
    gen_1 = containers.GenericExecutor(local_dictionary_1, local_input_dictionary_1)
    sh_1_f = containers.ShardTask(gen_1, "f", 0.001)
    sh_1_b = containers.ShardTask(gen_1, "b", 0.001)
    
    local_dictionary_2 = {idx: layer_dictionary[idx] for idx in range(20, 30)}
    local_input_dictionary_2 = {idx: io_dictionary[idx] for idx in range(20, 30)}
    gen_2 = containers.GenericExecutor(local_dictionary_2, local_input_dictionary_2)
    sh_2_f = containers.ShardTask(gen_2, "f", 0.001)
    sh_2_b = containers.ShardTask(gen_2, "b", 0.001)
    
    shard_dictionary = {
        0: sh_0_f,
        1: sh_1_f,
        2: sh_2_f,
        3: sh_2_b,
        4: sh_1_b,
        5: sh_0_b
    }
    
    shard_to_input_dict = {
        0: ["batch_0"],
        1: [9],
        2: [19],
        3: [19],
        4: [9],
        5: ["batch_0"]
    }
    
    shard_to_output_dict = {
        0: [1],
        1: [2],
        2: ["END"],
        3: ["END"],
        4: [2],
        5: [1]
    }
        
    mini_batch_time = 2.5
    
    model_task = ModelTask(name, layer_dictionary, io_dictionary, get_data_loader_train(32), 0.001, 4)
    model_task.setup(None, shard_dictionary, shard_to_input_dict, shard_to_output_dict, mini_batch_time)
    return model_task

def main():


    device_count = torch.cuda.device_count()

    model_0 = get_model("Model_0")
    model_1 = get_model("Model_1")
    model_2 = get_model("Model_2")
    
    # create orchestrator
    orchestra = ModelOrchestrator([model_0, model_1, model_2])
    orchestra.verbose = 1

    orchestra.train_models()
    
if __name__ == "__main__":
    main()
