
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
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from os import path
from timeit import timeit as timer


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# The model itself isn't huge, so we compensate with a *massive* batch size
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=False)




"""
    Modulized version of a .view function in PyTorch to support Hydra's module-to-module structure.
"""

class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_data):
        return in_data.view(in_data.size(0), -1)


fashionCNN = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    View(),
    nn.Linear(in_features=64*6*6, out_features=600),
    nn.Linear(in_features=600, out_features=300),
    nn.Dropout2d(0.25),
    nn.Linear(in_features=300, out_features=120),
    nn.Linear(in_features=120, out_features=60),
    nn.Linear(in_features=60, out_features=10)
    
)


"""
    Main function
"""

def test(model):
    with torch.no_grad():
        model.cpu()
        total = 0
        correct = 0
        for images, labels in test_dataloader:

            outputs = model(images)

            predictions = torch.max(outputs, 1)[1]
            correct += (predictions == labels).sum()

            total += len(labels)

        accuracy = correct * 100 / total
        print("Accuracy: {}".format(accuracy))
        

def main():


    device_count = torch.cuda.device_count()

    params = sum(p.numel() for p in fashionCNN.parameters())
    print("Total parameters: {}".format(params))
    #test(fashionCNN)
    lr_0 = 0.01
    
    epochs_0 = 10
    
    task_0 = ModelTask("Model 0", fashionCNN, nn.CrossEntropyLoss(), train_dataloader, lr_0, epochs_0)


    # create orchestrator
    orchestra = ModelOrchestrator([task_0])
    orchestra.verbose = 1

    """
     Double-Buffer size.
    """
    orchestra.buffer = 2000

    orchestra.generate()
    orchestra.train_models()
    test(fashionCNN)
if __name__ == "__main__":

    main()

