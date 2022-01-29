
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


import hydra
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
def main():


    device_count = torch.cuda.device_count()

    model_0 = torch.nn.Sequential(

        custom.BertEmbedding(28783, 1024, transpose=False),
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 4
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 8
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 12
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 16
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 20
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 24 - BERT-Large Size (340M params)
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 28
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 32
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 12
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 36
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 40
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 44
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 48
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 52
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 56
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 60
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 64
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 68
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 72
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 76
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 80
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 84
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 88
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 92
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 28
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 96
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 100
       
 
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.LayerNorm(1024, eps=1e-12),
        torch.nn.Linear(1024, 28783)

    )


    model_1 = copy.deepcopy(model_0)
    model_2 = copy.deepcopy(model_0)

    dataloader_0 = get_data_loader_train(32) # Generate dataloader

    dataloader_1 = get_data_loader_train(32)

    dataloader_2 = get_data_loader_train(64)
    
    lr_0 = 0.01
    lr_1 = 0.001
    lr_2 = 0.001
    
    epochs_0 = 2
    epochs_1 = 4
    epochs_2 = 8
    
    task_0 = ModelTask("Model 0", model_0, pretraining_loss, dataloader_0, lr_0, epochs_0)
    task_1 = ModelTask("Model 1", model_1, pretraining_loss, dataloader_1, lr_1, epochs_1)
    task_2 = ModelTask("Model 2", model_2, pretraining_loss, dataloader_2, lr_2, epochs_2)


    # create orchestrator
    orchestra = ModelOrchestrator([task_0, task_1, task_2])
    orchestra.verbose = 1

    """
     The Double-Buffer. Adjusting this up or down a bit can help to address minor
     errors in partitioning memory consumption.
    """
    orchestra.buffer = 10000

    orchestra.generate()
    orchestra.train_models()

if __name__ == "__main__":
    main()
