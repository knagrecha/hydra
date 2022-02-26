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
    Generic Backward Pass module. Any back-pass module
    must support these inputs (they can be discarded if unnecessary)
    and return a scaler (can be None), and gradients.
"""

class Backward():
    def __init__(self, idx):
        self.type="Backward"
        self.idx = idx

    def run(self, model, optimizer, batch_input, device, back_input, entry_point, scaler=None):
        
        model.to(device, non_blocking=True)
        model.zero_grad()  # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()  # zero the gradient buffers

        # Gradients
        if not isinstance(batch_input, torch.Tensor):
            batch_input = [x.to(device, non_blocking=True) for x in batch_input]
        else:
            batch_input = batch_input.to(device, non_blocking=True)
            
        # Backprop entry point
        if not isinstance(entry_point, torch.Tensor):
            entry_point = [x.to(device, non_blocking=True) for x in entry_point]
        else:
            entry_point = entry_point.to(device, non_blocking=True) 

        with torch.cuda.amp.autocast():
            torch.autograd.backward(entry_point, batch_input)

        del batch_input
        pass_back_gradients = None
        if self.idx != 0: # the first backwards pass need not compute back pass gradients.
            if (not isinstance(back_input, torch.Tensor)):
                pass_back_gradients = [i.grad for i in back_input]
            else:
                pass_back_gradients = back_input.grad             
            # the user will pass in what WAS the input for this stage!
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()

        if isinstance(back_input, list):
            while (len(back_input) > 0):
                del back_input[0]
            del back_input
        else:
            del back_input

        model.zero_grad()


        return scaler, pass_back_gradients 
