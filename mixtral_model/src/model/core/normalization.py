import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 eps: float=1e-6):
        super().__init__()

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '''
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms * self.weights

        return x_norm