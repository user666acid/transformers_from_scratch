import torch
import torch.nn as nn

from typing import Tuple, Optional

from ..core.normalization import RMSNorm
from ..core.mixture_of_experts import MoE
from ..core.positional_encoding import RoPE
from ..core.attention import GroupedQueryAttention

class Decoder(nn.Module):
    def __init__(self,
                 num_q_heads: int,
                 num_kv_heads: int,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int,
                 rope: RoPE, 
                 num_experts: int,
                 top_k_experts: int,
                 window_size: int,
                 dropout: float=0.1):
        """Блок декодера модели.

        Args:
            num_q_heads: Количество голов внимания для запросов.
            num_kv_heads: Количество голов внимания для ключей и значений.
            emb_size: Размерность внутреннего представления.
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            rope: Объект для слоя позиционного кодирования RoPE.
            num_experts: Общее количество экспертов.
            top_k_experts: Количество отбираемых экспертов.
            window_size: Длина окна внимания.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        
        self.multihead_attn = GroupedQueryAttention(num_q_heads, num_kv_heads, emb_size, head_size, max_seq_len, rope, window_size, dropout)
        self.moe = MoE(emb_size, num_experts, top_k_experts, dropout)
        
        self.norm_1 = RMSNorm(emb_size)
        self.norm_2 = RMSNorm(emb_size)
        
    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Определяет логику вычислений в блоке декодера.
        Используется pre-layer нормализация, GQA, остаточная связь и Mixture of Experts.

        Args:
            x: Исходное представление последовательности.
            use_cache: Флаг, контролирующий использование KV-кэша.
            cache: Содержит предпосчитанные матрицы ключей и значений.

        Returns:
            Преобразованное представление, KV-кэши.
        """
        attn_outputs, cache = self.multihead_attn(self.norm_1(x), use_cache, cache)
        
        x = x + attn_outputs
        x = x + self.moe(self.norm_2(x))

        return (x, cache)