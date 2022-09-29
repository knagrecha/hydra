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

from hydra.utilities import delete_batch, move_batch_to_device, track_gradients, untrack_gradients
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

    def run(self, model, optimizer, batch_input, device, back_input, scaler=None):
        model.to(device, non_blocking=True)
        toy_input = move_batch_to_device(back_input, device)
        
        if self.idx != 0:
            track_gradients(toy_input)
            
        batch_input = move_batch_to_device(batch_input, device)

        with torch.cpu.amp.autocast():
            toy_output = model(toy_input)
        
        if scaler is not None:
            toy_output = scaler.scale(toy_output)
            
        torch.autograd.backward(toy_output, batch_input)
        
        del toy_output, batch_input
        
        pass_back_gradients = None
        
        
        
        if self.idx != 0: # the first backwards pass need not compute back pass gradients.
            pass_back_gradients = untrack_gradients(toy_input)

                
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        delete_batch(toy_input)

        model.zero_grad(set_to_none=True)
        
        return scaler, pass_back_gradients 
