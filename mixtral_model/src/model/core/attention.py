import torch
import torch.nn as nn

from typing import Tuple, Optional

from .positional_encoding import RoPE

class GroupedQueryAttention(nn.Module):
    def __init__(self,
                 num_q_heads:int,
                 num_kv_heads: int,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int,
                 rope: RoPE,
                 window_size: int,
                 dropout: float=0.1):
        """Слой для GroupedQueryAttention.
        Вариация MHA, призванная повысить эффективность (в частности, уменьшить размер KV-кэша).
        Матрицы запросов уникальны для каждой головы внимания.
        Матрицы ключей и значений являются общими для каждой группы из num_q_heads // num_kv_heads голов внимания.
        Ограничение контекста внимания: для каждого токена внимание распространяется лишь на window_size токенов назад.

        Args:
            num_q_heads: Количество голов внимания для запросов.
            num_kv_heads: Количество голов внимания для ключей и значений.
            emb_size: Размерность внутреннего представления.
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            rope: Объект для слоя позиционного кодирования RoPE.
            window_size: Длина окна внимания.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.window_size = window_size

        self.rope = rope
        
        self.W_q = nn.Linear(emb_size, num_q_heads * head_size)
        self.W_k = nn.Linear(emb_size, num_kv_heads * head_size)
        self.W_v = nn.Linear(emb_size, num_kv_heads * head_size)
        
        self.mask = self.get_window_attention_mask(max_seq_len, window_size)
        
        self.W_o = nn.Linear(num_q_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def get_window_attention_mask(self,
                                  max_seq_len: int,
                                  window_size: int) -> torch.Tensor:
        """Получение маски для вычисления внимания с ограниченным контекстом.

        Args:
            max_seq_len: Максимальная длина последовательности.
            window_size: Длина окна внимания.

        Returns:
            Маска для матрицы значений внимания.
        """
        row_idxs = torch.arange(max_seq_len).unsqueeze(1)
        column_idxs = torch.arange(max_seq_len).unsqueeze(0)
        
        attention_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        window_mask = (row_idxs - column_idxs <= window_size).float()
        window_attention_mask = attention_mask * window_mask
        
        return window_attention_mask
        
    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     cache: Optinal[Tuple[torch.Tensor, torch.Tensor]]
                                     ) -> torch.Tensor:
        """Вычисление матрицы внимания.

        Args:
            Q: Матрица запросов.
            K: Матрица ключей.
            V: Матрица значений.
            cache: Содержит предпосчитанные матрицы ключей и значений.

        Returns:
            Взвешанные по вниманию значения.
        """
        _, _, seq_len, _ = Q.shape
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / self.head_size ** 0.5
        if not cache:
            mask = (self.mask[:seq_len, :seq_len] == 0)
            attn_scores = attn_scores.masked_fill(mask, float('-Inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_values = torch.matmul(attn_weights, V)

        return weighted_values
    
    def group_attention(self,
                        tensor: torch.Tensor,
                        group_size: int) -> torch.Tensor:
        """Условное приведение голов для матриц ключей и значений в размерность головы запросов.
        Выполняется путём виртуального (ссылающегося на то же место в памяти) копирования значений.

        Args:
            tensor: Исходная матрица.
            group_size: Размер группы.

        Returns:
            Матрица с продублированными значениями в 1-ом измерении.
        """
        batch_size, num_kv_heads, seq_len, head_size = tensor.size()
        dim = 1

        repeated_tensor = (
            tensor.unsqueeze(dim + 1)
            .expand(batch_size, num_kv_heads, group_size, seq_len, head_size)          
            .contiguous()
            .view(batch_size, num_kv_heads * group_size, seq_len, head_size)
        )
        
        return repeated_tensor

    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.
            use_cache: Флаг, контролирующий использование KV-кэша.
            cache: Содержит предпосчитанные матрицы ключей и значений.

        Returns:
            Преобразованное представление, KV-кэш.
        """
        batch_size, seq_len, _ = x.size()
        group_size = self.num_q_heads // self.num_kv_heads
        
        Q = self.W_q(x).reshape(batch_size, seq_len, self.num_q_heads, self.head_size).transpose(1, 2)
        K = self.W_k(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        V = self.W_v(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        
        if cache:
            start_pos = cache[0].size(2)
            Q_rotated = self.rope(Q, start_pos)
            K_rotated = self.rope(K, start_pos)
        else:
            Q_rotated = self.rope(Q)
            K_rotated = self.rope(K)
        
        if use_cache:
            if cache:
                K_rotated = torch.cat([cache[0], K_rotated], dim=2)
                V = torch.cat([cache[1], V], dim=2)
                         
            K_rotated_group = self.group_attention(K_rotated, group_size)
            V_group = self.group_attention(V, group_size)

            attn = self.scaled_dot_product_attention(Q_rotated, K_rotated_group, V_group, cache)
            cache = (K_rotated[:, :, -self.window_size:, :], V[:, :, -self.window_size:, :])    
        else:
            K_rotated_group = self.group_attention(K_rotated, group_size)
            V_group = self.group_attention(V, group_size)
            attn = self.scaled_dot_product_attention(Q_rotated, K_rotated_group, V_group, cache)
        
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.num_q_heads * self.head_size)
        attn = self.W_o(attn)
        attn = self.dropout(attn)
        
        return (attn, cache)