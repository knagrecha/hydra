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
    The "shard" wrapper. Holds reference to a 'model' - an NNContainer consisting
    of a subset of layers of the original model.
    
    Also holds a "direction", determining which microbatch-stage this shard covers.
    
    Holds reference to a learning rate, though we can replace this with the ModelTask LR.
    
    The optimizer per-shard is currently just SGD, but we will extend to Adam in the future.

"""

class ShardedTask():

    def __init__(self, model, executor, direction, time_taken, idx, lr, *args):
        self.lr = lr
        self.model = model
        for param in self.model.parameters():
            param.pin_memory()
        self.direction = direction
        self.time_cost = time_taken
        self.idx = idx
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        self.executor = executor
    
    def run(self, arg_list):
        if self.executor.type != "Forward":
            return self.executor.run(self.model, self.optimizer, *arg_list)
        else:
            return self.executor.run(self.model, *arg_list)
