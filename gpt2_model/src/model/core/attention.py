import torch
import torch.nn as nn

from typing import Optional, Tuple 

class HeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int):
        super().__init__()

        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)

        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        '''
        '''
        K = self.W_k(x)
        Q = self.W_q(x)
        V = self.W_v(x)
        
        if use_cache:
            if cache:
                K = torch.cat([cache[0], K], dim=1)
                V = torch.cat([cache[1], V], dim=1)

            attn = self.scaled_dot_product_attention(Q, K, V, cache)
            cache = (K, V)
        else:
            attn = self.scaled_dot_product_attention(Q, K, V, cache)
        
        return (attn, cache)

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     cache: Optional[Tuple[torch.Tensor, torch.Tensor]]
                                     ) -> torch.Tensor:
        '''
        '''
        _, seq_len, _ = Q.shape
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / self.head_size ** 0.5
        
        if not cache:
            mask = (self.mask[:seq_len, :seq_len] == 0)
            attn_scores = attn_scores.masked_fill(mask, float('-Inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_values = torch.matmul(attn_weights, V)

        return weighted_values

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads:int,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int,
                 dropout: float=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.attn_heads = nn.ModuleList([HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)])
        self.W_o = nn.Linear(head_size * num_heads, emb_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        '''
        '''
        if use_cache:
            attn_outputs = []
            current_cache = []
            for i, head in enumerate(self.attn_heads):    
                head_cache = cache[i] if cache else None
                attn_output, head_cache_new = head(x, use_cache, head_cache)

                attn_outputs.append(attn_output)
                current_cache.append(head_cache_new)
            
            attn_outputs = torch.cat(attn_outputs, dim=-1)
            cache = current_cache

        else:
            attn_outputs = torch.cat([head(x, use_cache, cache)[0] for head in self.attn_heads], dim=-1)

        attn_outputs = self.W_o(attn_outputs)
        attn_outputs = self.dropout(attn_outputs)
        
        return (attn_outputs, cache)