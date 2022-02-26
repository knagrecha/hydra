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

from hydra.utilities import delete_batch, move_batch_to_device
import torch

"""
    Forward pass at the end of the model. Must accept as input
    model, optimizer, data batch, labels, a criterion, the device,
    and scaler (optional).

"""

class ForwardLoss():
    def __init__(self, idx):
        self.type="Forward Loss"
        self.idx = idx

    def run(self, model, optimizer, batch_input, labels, criterion, device, scaler=None):
        
        old = next(model.parameters()).device
        model.to(device, non_blocking=True)

        batch_input = move_batch_to_device(batch_input, device)
        
        model.zero_grad()  # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()  # zero the gradient buffers
        
        labels = move_batch_to_device(labels, device)
        

        if self.idx != 0:
            if not isinstance(batch_input, torch.Tensor):
                for batch in batch_input:
                    batch.requires_grad_(True)
            else:
                batch_input.requires_grad_(True)


        with torch.cuda.amp.autocast():
            ns_labels = model(batch_input)
            loss = criterion(ns_labels, labels)

        if (scaler is not None):
            loss = scaler.scale(loss)

        loss.backward()


        pass_back_gradients = []
        
        if self.idx != 0:
            # pass_back_gradients are on device
            if not isinstance(batch_input, torch.Tensor):
                pass_back_gradients = [ batch.grad for batch in batch_input ]
            else:
                pass_back_gradients.append(batch_input.grad)
        else:
            pass_back_gradients = None

        if (scaler is not None):
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()

        model.zero_grad()

        #shard_model = shard.model.to("cpu", non_blocking=True)

        del labels


        return scaler, pass_back_gradients, loss.item()
