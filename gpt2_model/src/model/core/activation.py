import torch
import torch.nn as nn

import numpy as np

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '''
        tanh_value = torch.tanh(torch.sqrt(torch.as_tensor(2 / np.pi)) * (x + 0.044715 * x ** 3))
        activation = 0.5 * x * (1 + tanh_value)

        return activation