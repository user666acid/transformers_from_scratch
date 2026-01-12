import torch
import torch.nn as nn

class PositionalEmbeddings(nn.Module):
    def __init__(self,
                 max_seq_len: int,
                 emb_size: int):
        """Слой для кодирования позиций элементов последовательности.
         Используется обучаемое представление.

        Args:
            max_seq_len: Максимальная длина последовательности.
            emb_size: Размерность внутреннего представления.
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        
        self.pos_embeddings = nn.Embedding(max_seq_len, emb_size)

    def forward(self,
                seq_len: int,
                start_pos: int=0) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            seq_len: Длина обрабатываемой последовательности.

        Returns:
            Представление релевантных позиций.
        """
        pos_embeddings = self.pos_embeddings.weight[start_pos: start_pos + seq_len, :]
        
        return pos_embeddings