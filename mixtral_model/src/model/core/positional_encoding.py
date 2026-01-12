import torch
import torch.nn as nn

from typing import Tuple

class RoPE(nn.Module):
    def __init__(self,
                 head_size: int,
                 max_seq_len: int,
                 base: int=10000):
        """Слой для кодирование относительных позиций элементов последовательности через Rotary Positional Embeddings.

        Args:
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            base: Константа для расчёта частоты вращения.
        """
        super().__init__()

        cos_matrix, sin_matrix = self.get_angles_matrices(head_size, max_seq_len, base)
        self.register_buffer('cos_matrix', cos_matrix)
        self.register_buffer('sin_matrix', sin_matrix)

    def get_angles_matrices(self,
                            head_size: int,
                            max_seq_len: int,
                            base: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вычисление матриц для косинусов и синусов углов вращения.

        Args:
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            base: Константа для расчёта частоты вращения.

        Returns:
            Матрицы для косинусов и синусов углов вращения.
        """
        d = head_size // 2
        exp = 2 * torch.arange(d, dtype=torch.float32) / head_size
        inverse_freq = 1 / torch.pow(base, exp)
        pos_idxs = torch.arange(max_seq_len, dtype=torch.float32)

        freq_matrix = pos_idxs.unsqueeze(1) @ inverse_freq.unsqueeze(0)
        cos_matrix = torch.cos(freq_matrix)
        sin_matrix = torch.sin(freq_matrix)

        return (cos_matrix, sin_matrix)


    def forward(self,
                x: torch.Tensor,
                start_pos: int=0) -> torch.Tensor:
        """Определяет логику вычислений в слое.
        Выполняет вращение векторов в двумерном (на уровне пар элементов) пространстве с помощью матрицы вращения.

        Args:
            x: Исходное представление последовательности.
            start_pos: Индекс начальной позиции, необходим при использовании KV-кэша.

        Returns:
            Представление последовательности после выполнения вращения элементов.
        """
        _, _, seq_len, _ = x.size()

        cos = self.cos_matrix[start_pos: start_pos + seq_len, :]
        sin = self.sin_matrix[start_pos: start_pos + seq_len, :]

        x_even = x[:, :, :, ::2]
        x_odd = x[:, :, :, 1::2]
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_odd * cos + x_even * sin
        
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, :, ::2] = x_even_rotated
        x_rotated[:, :, :, 1::2] = x_odd_rotated

        return x_rotated