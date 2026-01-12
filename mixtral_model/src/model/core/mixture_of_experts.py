import torch
import torch.nn as nn

class SiLU(nn.Module):
    def __init__(self):
        """Слой для функции активации Sigmoid Linear Unit.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Активированное представление.
        """
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float=0.1):
        """Слой для функции активации Swish Gated Linear Unit.

        Args:
            emb_size: Размерность внутреннего представления.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()

        self.gate = nn.Linear(emb_size, 4 * emb_size)
        self.up = nn.Linear(emb_size, 4 * emb_size)
        self.down = nn.Linear(4 * emb_size, emb_size)

        self.activation = SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Активированное представление.
        """
        activation = self.down(self.up(x) * self.activation(self.gate(x)))
        activation = self.dropout(activation)

        return activation

class MoE(nn.Module):
    def __init__(self,
                 emb_size: int,
                 num_experts: int,
                 top_k_experts: int,
                 dropout: float=0.1):
        """Слой для модуля Mixture of Experts. Представляет собой развитие FFN-слоя: вместо одного FFN
        каждый токен независимо пропускается через top-k (одна структура, но разные веса) сетей-экспертов (также FFN).
        Выбор и взвешивание экспертов происходит в router-сети.

        Args:
            emb_size: Размерность внутреннего представления.
            num_experts: Общее количество экспертов.
            top_k_experts: Количество отбираемых экспертов.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()

        self.top_k_experts = top_k_experts
        self.num_experts = num_experts

        self.router = nn.Linear(emb_size, num_experts)
        self.experts = nn.ModuleList([SwiGLU(emb_size, dropout) for _ in range(num_experts)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Для каждого токена определяются top-k экспертов и их веса.
        Далее в цикле по всем экспертам:
        - Отбор релевантных для рассматриваемого эксперта токенов (тех, для которых данный эксперт попал в top-k)
        - Токены пропускаются через сеть-эксперта, выходы взвешиваются
        - Результат прибавляется к выходу слоя

        Args:
            x: Исходное представление последовательности.

        Returns:
            MOE-активированное представление.
        """
        _, _, emb_size = x.size()

        router_logits = self.router(x)
        top_k_logits, selected_experts = router_logits.topk(self.top_k_experts, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        sparse_weights = torch.full_like(router_logits, 0)
        sparse_weights = sparse_weights.scatter(-1, selected_experts, top_k_weights)

        moe_out = torch.zeros_like(x)
        x_flat = x.view(-1, emb_size)
        weights_flat = sparse_weights.view(-1, self.num_experts)

        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = x_flat[flat_mask]
                expert_output = expert(expert_input)

                expert_weight = weights_flat[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * expert_weight

                moe_out[expert_mask] += weighted_output.squeeze(1)
        
        moe_out = self.dropout(moe_out)
        
        return moe_out