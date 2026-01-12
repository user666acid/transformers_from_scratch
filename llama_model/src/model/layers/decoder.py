import torch
import torch.nn as nn

from typing import Tuple, Optional

from ..core.activation import SwiGLU
from ..core.normalization import RMSNorm
from ..core.positional_encoding import RoPE
from ..core.attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self,
                 num_heads: int,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int,
                 rope: RoPE, 
                 dropout: float=0.1):
        """Блок декодера модели.

        Args:
            num_heads: Количество голов внимания.
            emb_size: Размерность внутреннего представления.
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            rope: Объект для слоя позиционного кодирования RoPE.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        
        self.multihead_attn = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, rope, dropout)
        self.ffn = SwiGLU(emb_size, dropout)
        
        self.norm_1 = RMSNorm(emb_size)
        self.norm_2 = RMSNorm(emb_size)
        
    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Определяет логику вычислений в блоке декодера.
        Используется pre-layer нормализация, RoPE, MHA с KV-кэшированием, остаточная связь и полносвязный слой.

        Args:
            x: Исходное представление последовательности.
            use_cache: Флаг, контролирующий использование KV-кэша.
            cache: Содержит предпосчитанные матрицы ключей и значений.

        Returns:
            Преобразованное представление, KV-кэши.
        """
        attn_outputs, cache = self.multihead_attn(self.norm_1(x), use_cache, cache)
        
        x = x + attn_outputs
        x = x + self.ffn(self.norm_2(x))

        return (x, cache)