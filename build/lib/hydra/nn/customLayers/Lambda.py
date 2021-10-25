import torch
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()
        self.name = "select index 0"

