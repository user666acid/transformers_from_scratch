import torch
import torch.nn as nn

class PositionalEmbeddings(nn.Module):
    def __init__(self,
                 max_seq_len: int,
                 emb_size: int):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        
        self.pos_embeddings = nn.Embedding(max_seq_len, emb_size)

    def forward(self,
                seq_len: int,
                start_pos: int=0) -> torch.Tensor:
        '''
        '''
        pos_embeddings = self.pos_embeddings.weight[start_pos: start_pos + seq_len, :]
        
        return pos_embeddings