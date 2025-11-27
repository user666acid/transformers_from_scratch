import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '''
        K = self.W_k(x)
        Q = self.W_q(x)
        V = self.W_v(x)

        attn = self.scaled_dot_product_attention(Q, K, V)

        return attn

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor) -> torch.Tensor:
        '''
        '''
        _, seq_len, _ = Q.shape
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / self.head_size ** 0.5
        
        mask = (self.mask[:seq_len, :seq_len] == 0)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        '''
        '''     
        attn_outputs = torch.cat([head(x) for head in self.attn_heads], dim=-1)
        attn_outputs = self.W_o(attn_outputs)
        attn_outputs = self.dropout(attn_outputs)
        
        return attn_outputs