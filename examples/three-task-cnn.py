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
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from os import path
import torch.nn as nn
from timeit import timeit as timer
from einops import rearrange, repeat
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange




"""
    Helper function to create a training dataloader.
"""

def get_data_loader_train(b_size):
    start = timer()
    print("\nPreparing to load dataset....")

    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = FakeData(size=100, image_size=(3, 500, 500), num_classes=100, transform=transform)

    return torch.utils.data.DataLoader(dataset, batch_size=b_size)


    
def get_model(depth):
    modules = []
    
    for i in range(depth):
        modules.append(torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding='same'))

    modules.append(nn.Flatten())
    modules.append(nn.Linear(750000, 100))

    return nn.Sequential(*modules)
 
"""
    Main function
"""
def main():
    
    model_0 = get_model(24) 
    model_1 = get_model(36) # 800M params
    model_2 = get_model(48)
    
    params = sum(p.numel() for p in model_0.parameters())
    print("Total parameters: {}".format(params))

    task_0 = ModelTask("Model 0", model_0, nn.CrossEntropyLoss(), get_data_loader_train(128), 0.001, 5)    
    task_1 = ModelTask("Model 1", model_1, nn.CrossEntropyLoss(), get_data_loader_train(128), 0.001, 5)
    task_2 = ModelTask("Model 2", model_2, nn.CrossEntropyLoss(), get_data_loader_train(128), 0.001, 5)
    
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
