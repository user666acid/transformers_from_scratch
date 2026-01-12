import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 eps: float=1e-6):
        """Слой для Root Mean Square нормализации внутреннего представления.
        Упрощение LayerNorm: отказ от центрирования при сохранении масштабирования.

        Args:
            dim: Размерность нормализации.
            eps: Константа для стабильности вычислений.
        """
        super().__init__()

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Нормализованное представление.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms * self.weights

        return x_norm