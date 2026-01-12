import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_size: int):
        """Слой для получение исходного внутреннего представления.

        Args:
            vocab_size: Размерность словаря модели.
            emb_size: Размерность внутреннего представления.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        
        self.embeddings = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходная последовательность токенов.

        Returns:
            Исходное внутреннее представление.
        """
        x = self.embeddings(x)

        return x