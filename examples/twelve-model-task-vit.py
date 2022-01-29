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


# This task requres A LOT of DRAM, so be careful!

import hydra
from hydra import ModelTask, ModelOrchestrator
import customLayers as custom
import copy
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from os import path
import torch.nn as nn
from timeit import timeit as timer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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


    dataset = CIFAR10("cifar", train=True, download=True, transform=transform)

    return torch.utils.data.DataLoader(dataset, batch_size=b_size)





# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# This will be treated as one single layer
class Prepper(nn.Module):
    def __init__(self, image_size, patch_size, dim, emb_dropout, channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
    
class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Identity()
    def forward(self, x):
        return self.cls(x[:, 0])
    
def get_model(depth, hidden=1280, mlp_size=5120, heads=16):
    modules = []
    modules.append(Prepper(image_size=32, patch_size=8, channels=3, dim=hidden, emb_dropout=0.5))
    
    for i in range(depth):
        modules.append(Residual(PreNorm(hidden, Attention(dim=hidden, heads=heads, dim_head=64, dropout =0.5))))
        modules.append(Residual(PreNorm(hidden, FeedForward(hidden_dim=hidden, dim=mlp_size, dropout =0.5))))
    modules.append(Selector()),
    modules.append(nn.Linear(1280, 10))

    return nn.Sequential(*modules)
 
"""
    Main function
"""
def main():
    
    model_0 = get_model(24, 1024, 4096, 16) # Vit-Large size (300M parms)
    model_1 = get_model(24, 1024, 4096, 16) 
    model_2 = get_model(32)# ViT-Huge Size (600M params)
    model_3 = get_model(32) 
    model_4 = get_model(42) # 800M params
    model_5 = get_model(42)
    model_6 = get_model(54) # 1B params
    model_7 = get_model(54)
    model_8 = get_model(80) # 1.5B params
    model_9 = get_model(80)
    model_10 = get_model(110) # 2B params
    model_11 = get_model(110)
    
    params = sum(p.numel() for p in model_11.parameters())
    print("Total parameters: {}".format(params))


    task_0 = ModelTask("Model 0", model_0, pretraining_loss, get_data_loader_train(64), 0.001, 5)
    task_1 = ModelTask("Model 1", model_1, pretraining_loss, get_data_loader_train(128), 0.001, 5)

    task_2 = ModelTask("Model 2", model_2, pretraining_loss, get_data_loader_train(64), 0.001, 5)
    task_3 = ModelTask("Model 3", model_3, pretraining_loss, get_data_loader_train(128), 0.001, 5)
    
    task_4 = ModelTask("Model 4", model_4, pretraining_loss, get_data_loader_train(64), 0.001, 5)
    task_5 = ModelTask("Model 5", model_5, pretraining_loss, get_data_loader_train(128), 0.001, 5)
    
    task_6 = ModelTask("Model 6", model_6, pretraining_loss, get_data_loader_train(64), 0.001, 5)
    task_7 = ModelTask("Model 7", model_7, pretraining_loss, get_data_loader_train(128), 0.001, 5)
    
    task_8 = ModelTask("Model 8", model_8, pretraining_loss, get_data_loader_train(64), 0.001, 5)
    task_9 = ModelTask("Model 9", model_9, pretraining_loss, get_data_loader_train(128), 0.001, 5)
    
    task_10 = ModelTask("Model 10", model_10, pretraining_loss, get_data_loader_train(64), 0.001, 5)
    task_11 = ModelTask("Model 11", model_11, pretraining_loss, get_data_loader_train(128), 0.001, 5)

    

    # create orchestrator
    orchestra = ModelOrchestrator([task_0, task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10, task_11])
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
