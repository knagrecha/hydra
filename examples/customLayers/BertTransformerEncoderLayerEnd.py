import torch.nn as nn
import torch.nn.functional as F
import torch

class BertTransformerEncoderLayerEnd(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(BertTransformerEncoderLayerEnd, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input):
        #print(input.shape)
        input = input[:, 0, :]
        return F.linear(input, self.weight, self.bias)
