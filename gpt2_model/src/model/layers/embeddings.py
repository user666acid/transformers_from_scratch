import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        
        self.embeddings = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        '''
        x = self.embeddings(x)

        return x