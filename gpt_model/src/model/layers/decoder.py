import torch
import torch.nn as nn

from ..core.feedforward import FeedForward
from ..core.attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self,
                 num_heads: int,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int,
                 dropout: float=0.1):
        """Блок декодера модели.

        Args:
            num_heads: Количество голов внимания.
            emb_size: Размерность внутреннего представления.
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        
        self.multihead_attn = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.ffn = FeedForward(emb_size, dropout)
        
        self.norm_1 = nn.LayerNorm(emb_size)
        self.norm_2 = nn.LayerNorm(emb_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в блоке декодера.
        Используется post-layer нормализация, MHA, остаточная связь и полносвязный слой.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Преобразованное представление.
        """
        x = self.norm_1(x + self.multihead_attn(x))
        x = self.norm_2(x + self.ffn(x))
        
        return x