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

    def run(self, model, optimizer, batch_input, device, back_input, scaler=None):
        
        model.to(device, non_blocking=True)
        if not isinstance(back_input, torch.Tensor):
            toy_input = [x.to(device, non_blocking=True) for x in back_input]
            if self.idx != 0:
                for m_input in toy_input:
                    if isinstance(m_input, torch.Tensor):
                        m_input.requires_grad_(True)

        else:
            toy_input = back_input.to(device, non_blocking=True)
            if self.idx != 0:
                toy_input.requires_grad_(True)     

        if not isinstance(batch_input, torch.Tensor):
            batch_input = [x.to(device, non_blocking=True) for x in batch_input]
        else:
            batch_input = batch_input.to(device, non_blocking=True) 


        with torch.cuda.amp.autocast():
            toy_output = model(toy_input)
        
        if scaler is not None:
            toy_output = scaler.scale(toy_output)
        torch.autograd.backward(toy_output, batch_input)
        del toy_output
        del batch_input
        pass_back_gradients = None
        if self.idx != 0: # the first backwards pass need not compute back pass gradients.
            if (not isinstance(toy_input, torch.Tensor)):
                pass_back_gradients = [i.grad for i in toy_input]
                for m_input in toy_input:
                    m_input.requires_grad_(False)
            else:
                pass_back_gradients = toy_input.grad
                toy_input.requires_grad_(False)
            # the user will pass in what WAS the input for this stage!
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        if not isinstance(toy_input, torch.Tensor):
            while (len(toy_input) > 0):
                del toy_input[0]
            del toy_input
        else:
            del toy_input

        model.zero_grad(set_to_none=True)


        return scaler, pass_back_gradients 
