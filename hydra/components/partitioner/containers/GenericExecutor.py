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

import torch.nn as nn
import copy
import gc

import torch
"""
    Dictionary-defined executor. Enables Hydra to train 
    arbitrary computational DAGs.
    
    layer dictionary matches layer indices to layers.
    output_dictionary matches input provider indices to a list of layers (forward).
        Includes immediate connected recipients from the next shards down.
        DOES NOT INCLUDE the initial connectors of this shard (i.e. this shard's link to prior shards)
"""


class GenericExecutor(nn.Module):
    def __init__(self, layer_dictionary, input_dictionary, requested_outputs):
        super(GenericExecutor, self).__init__()
        
        self.layer_dictionary = layer_dictionary
        self.all_layers = self.layer_dictionary.keys()
        self.requested_outputs = set(requested_outputs) # maps input layers to the layers that receive it
        self.input_dictionary = input_dictionary
        
        for key, value in self.layer_dictionary.items():
            if (isinstance(value, nn.Module)):
                self.add_module("Module_{}".format(key), value)

    """
        Forward pass will continue to run until no internal layer-input match can be found.
        Tensor dictionary matches previous layer indices to their outputs.
        This is basically a 'BFS' through the model.
    """
    def forward_helper(self, tensor_dictionary):
        successful_pass = False
        while not successful_pass:
            # for every received provider, tensor
            update_dictionary = {}
            candidate_layers = {layer for layer in self.all_layers if all(req in tensor_dictionary for req in self.input_dictionary[layer]) } - tensor_dictionary.keys()
            for layer in candidate_layers:
                requested_inputs = [ tensor_dictionary[req] for req in self.input_dictionary[layer] ]
                tensor_dictionary[layer] = self.layer_dictionary[layer](*requested_inputs) # plug in all requested inputs

            if (len(candidate_layers) == 0): # if all recipients in this shard were exhausted
                successful_pass = True

        # clear out unneeded inputs
        remove_keys = tensor_dictionary.keys() - self.requested_outputs
        for key in remove_keys:
            tensor_dictionary.pop(key, None)

        return tensor_dictionary
        
    def forward(self, tensor_dictionary, no_grad=True):
        if no_grad:
            with torch.no_grad():
                return self.forward_helper(tensor_dictionary)
        else:
            return self.forward_helper(tensor_dictionary)
            
            
    """
        The actual backward pass DFS's to find leaf tensors before calling autograd 
        and updating gradient pass backs.
    """  
    
    
    
    
    def backward(self, in_tensor_dict, grad_tensor_dict): 
        saved_in = copy.copy(in_tensor_dict)
        successful_pass = False
        # if not the initial batch, we need gradients to pass back
        for key, value in in_tensor_dict.items():
            if not ( isinstance(key, str)  ):
                value.requires_grad_(True)
        
        # run forward, grad enabled
        output_dict = self.forward(in_tensor_dict, no_grad=False)
        
        
        # get the gradients and outputs
        ret_grads = [grad_tensor_dict[idx] for idx in self.requested_outputs]
        ret_outs = [output_dict[idx] for idx in self.requested_outputs]   
        
        torch.autograd.backward(ret_outs, ret_grads) # backprop
        
        new_dict = {}
        # record gradients if available
        for key, value in saved_in.items():
            if not ( isinstance(key, str) ):
                new_dict[key] = value.grad # define the gradient to be passed back
      
        
        return new_dict, ret_outs
