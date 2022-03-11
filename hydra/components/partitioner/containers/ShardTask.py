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


import torch

"""
    The "shard-stage" wrapper. Essentially refers to a particular shard's part of the minibatch 
    (known as a microbatch in prior art). 
    
    Holds reference to a 'model' of type GenericExecutor consisting
    of a subset of the original model dictionary.
    
    Also holds a "direction", determining which microbatch-stage this shard covers.
    
    Holds reference to a learning rate, though we can replace this with the ModelTask LR.
    
    The optimizer per-shard is currently just SGD, but we will extend to Adam in the future.

"""

class ShardTask():

    def __init__(self, model, executor, direction, time_taken, idx, lr):
        self.lr = lr
        self.model = model
        self.direction = direction
        self.time_cost = time_taken
        self.idx = idx
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        self.executor = executor
    
    def run(self, tensor_dictionary, gradient_tensor_dictionary=None):
        if self.direction == "f":
            self.model.forward(tensor_dictionary)
        else:
            self.model.backward(tensor_dictionary, gradient_tensor_dictionary)
            self.optimizer.step()