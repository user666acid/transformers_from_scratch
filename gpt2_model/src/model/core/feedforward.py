import torch
import torch.nn as nn

from .activation import GELU

class FeedForward(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float=0.1):
        super().__init__()

        self.emb_size = emb_size
  
        self.linear_wide = nn.Linear(emb_size, 4 * emb_size)
        self.linear_narrow = nn.Linear(4 * emb_size, emb_size)
        
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '''
        x = self.linear_wide(x)
        x = self.activation(x)
        x = self.linear_narrow(x)
        x = self.dropout(x)
        
        return x