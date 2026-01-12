import torch
import torch.nn as nn

from .activation import GELU

class FeedForward(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float=0.1):
        """Слой для FFN.

        Args:
            emb_size: Размерность внутреннего представления.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()

        self.emb_size = emb_size
  
        self.linear_wide = nn.Linear(emb_size, 4 * emb_size)
        self.linear_narrow = nn.Linear(4 * emb_size, emb_size)
        
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.
        Увеличение размерности представления -> активация -> уменьшение размерности -> дропаут.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Преобразованное представление.
        """
        x = self.linear_wide(x)
        x = self.activation(x)
        x = self.linear_narrow(x)
        x = self.dropout(x)
        
        return x