
import torch.nn as nn
from .utilities import get_free_space

import gc

import torch
class NNContainer(nn.Module):
    def __init__(self, layers):

        super(NNContainer, self).__init__()
        
        for idx, layer in enumerate(layers):
            self.add_module("Module_{}".format(idx), layer)

    def forward(self, x):
        for idx, mod in enumerate(self.children()):
            #print("IDX: {}, MODULE: {}".format(idx, mod))
            if not (isinstance(mod, NNContainer)):
                if (isinstance(x, tuple) or isinstance(x, list)):
                    #print (mod)
                    x = mod(*x)
                else:
                    x = mod(x)
                

        return x