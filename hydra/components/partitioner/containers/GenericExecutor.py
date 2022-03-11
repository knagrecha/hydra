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
from .utilities import get_free_space

import gc

import torch
"""
    Dictionary-defined executor. Enables Hydra to train 
    arbitrary computational DAGs.
    
    layer dictionary matches layer indices to layers.
    input_dictionary matches input provider indices to a list of layers (forward).
    reverse_input_dictionary dictionary recipient layer indices to provider indices.
        Includes immediate connected recipients from the next shards down.
"""


class GenericExecutor(nn.Module):
    def __init__(self, layer_dictionary, input_dictionary, reverse_input_dictionary):
        super(GenericExecutor, self).__init__()
        
        self.layer_dictionary = layer_dictionary
        self.input_dictionary = input_dictionary
        self.reverse_input_dictionary = reverse_input_dictionary
        
        for key, value in layer_dictionary:
            self.add_module("Module_{}".format(key), value)

    """
        Forward pass will continue to run until no internal layer-input match can be found.
        Tensor dictionary matches previous layer indices to their outputs.
    """
    def forward(self, tensor_dictionary):
        with torch.no_grad():
            successful_pass = False
            while not successful_pass:
                marked_for_deletion = []
                # for every received provider, tensor
                for key, value in tensor_dictionary:
                    if key in in self.input_dictionary: # check if the provider has a 
                                                        # corresponding recipient layer in this model
                        recipient_layers = self.input_dictionary[key] # identify receiving layers
                        for layer in recipient_layers:
                            tensor_dictionary[layer] = self.layer_dictionary[layer](value) # run execution
                        marked_for_deletion.append(key) # mark used tensor for deletion
                if (len(marked_for_deletion) == 0): # if all recipiients in this shard were exhausted
                    successful_pass = True
                else:
                    for key in marked_for_deletion:
                        del tensor_dictionary[key]


            return tensor_dictionary
    
    
    """
        Backward pass makes use of gradient checkpointing to regenerate tensors.
        Grad tensor dict matches layers to the gradients they should receive.
    """
    def backward(self, in_tensor_dict, grad_tensor_dict):
        successful_pass = False
        while not successful_pass:
            marked_for_deletion = []
            # for every received provider, tensor
            for key, value in in_tensor_dict: 
                if key in in self.input_dictionary: # check if the provider has a 
                                                    # corresponding recipient layer in this model
                    recipient_layers = self.input_dictionary[key]
                    for layer in recipient_layers:
                        in_tensor_dict[layer] = self.layer_dictionary[layer](value)
                    marked_for_deletion.append(key)
            if (len(marked_for_deletion) == 0):
                successful_pass = True
            else:
                for key in marked_for_deletion:
                    del in_tensor_dict[key]
                    
        
        # in_tensor_dict now ONLY contains our local shard outputs.
        # We will do the same dict-lookup pass as the forward pass, in reverse
        
        successful_pass = False
        while not successful_pass:
            marked_for_deletion = []
            for key, value in in_tensor_dict:
                if key in in self.input_dictionary:
                    grads_to_use = grad_tensor_dict[key]
                    for layer in recipient_layers:
                        in_tensor_dict[layer] = self.layer_dictionary[layer](value)
                    marked_for_deletion.append(key)
            if (len(marked_for_deletion) == 0):
                successful_pass = True
            else:
                for key in marked_for_deletion:
                    del in_tensor_dict[key]
                
                
        return tensor_dictionary
