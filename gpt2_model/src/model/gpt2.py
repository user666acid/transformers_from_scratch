import torch
import torch.nn as nn

from typing import List, Tuple, Optional

from .layers.decoder import Decoder
from .layers.embeddings import TokenEmbeddings
from .core.positional_encoding import PositionalEmbeddings

class GPT2(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 max_seq_len: int,
                 emb_size: int,
                 num_heads: int,
                 head_size: int,
                 num_layers: int,
                 dropout: float=0.1,
                 device: str='cpu'):
        """Модель GPT2

        Args:
            vocab_size: Размерность словаря модели.
            max_seq_len: Максимальная длина последовательности.
            emb_size: Размерность внутреннего представления.
            num_heads: Количество голов внимания.
            head_size: Размерность головы внимания.
            num_layers: Количество декодеров.
            dropout: Доля зануляемых элементов.
            device: Где хранить и совершать вычисления.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers 
        self.dropout = dropout
        self.device = device 
        
        self.token_emb = TokenEmbeddings(vocab_size, emb_size)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size)
        
        self.decoder = nn.Sequential(
            *[Decoder(num_heads, emb_size, head_size, max_seq_len) for _ in range(num_layers)]
        )

        self.proj = nn.Linear(emb_size, vocab_size)
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None
                ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Определяет логику вычислений в модели.

        Args:
            x: Исходная последовательность токенов.
            use_cache: Флаг, контролирующий использование KV-кэша.
            cache: Содержит предпосчитанные матрицы ключей и значений для декодеров.

        Returns:
            Логиты, предпосчитанные матрицы ключей и значений для декодеров.
        """
        seq_len = x.size()[1] if len(x.size()) == 2 else 1

        if use_cache:
            final_caches = []

            if cache:
                start_pos = cache[0][0][0].size(1)
                emb = self.dropout(self.token_emb(x) + self.pos_emb(seq_len=1, start_pos=start_pos))
                for i, decoder_layer in enumerate(self.decoder):
                    current_cache = cache[i]
                    emb, layer_cache = decoder_layer(emb, use_cache, current_cache)
                    final_caches.append(layer_cache)
            else:
                emb = self.dropout(self.token_emb(x) + self.pos_emb(seq_len))
                current_cache = None
                for decoder_layer in self.decoder:
                    emb, layer_cache = decoder_layer(emb, use_cache, current_cache)
                    final_caches.append(layer_cache)

            cache = final_caches

        else:
            emb = self.dropout(self.token_emb(x) + self.pos_emb(seq_len))
            for decoder_layer in self.decoder: 
                emb, _ = decoder_layer(emb, use_cache)
            

        emb = self.norm(emb)
        logits = self.proj(emb)

        return (logits, cache)

    def generate(self,
                 x: torch.Tensor,
                 max_new_tokens: int,
                 do_sample: bool,
                 temperature: float=1.0,
                 top_k: Optional[int]=None,
                 top_p: Optional[float]=None,
                 use_cache: bool=True) -> torch.Tensor:
        """Определяет логику генерации токенов.

        Args:
            x: Исходная последовательность токенов.
            max_new_tokens: Ограничение на максимальное количество сгенерированных токенов.
            do_sample: Флаг, контролирующий использование сэмплинга при генерации.
            temperature: Температура. Константа для масштабирования логитов, используется для контроля формы генерируемого распределния.
            top_k: Количество претендентов для top-k сэмплирования.
            top_p: Вероятностная масса для top-p сэмплирования.
            use_cache: Флаг, контролирующий использование KV-кэша.

        Returns:
            Сгенерированные токены.
        """
        cache = None
        for step in range(max_new_tokens):
            if use_cache and step == 0:
                x_in = x
            elif use_cache and step != 0:
                x_in = x[:, -1:]
            else:
                x_in = x[:, -self.max_seq_len:]

            logits, cache = self.forward(x_in, use_cache, cache)
            logits_last = logits[:, -1, :] / (temperature + 1e-10)

            if do_sample:
                logits_sampled = self.sample_logits(logits_last, top_k, top_p)
                probs = torch.softmax(logits_sampled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                probs = torch.softmax(logits_last, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)      

            x = torch.cat([x, next_token], dim=1)

        return x

    def sample_logits(self,
                      logits: torch.Tensor,
                      top_k: Optional[int],
                      top_p: Optional[float]) -> torch.Tensor:
        """Сэмплирование логитов.
        Отсеивает нерелевантные логиты в зависимости от выбранной стратегии.
        Отсеянным логитам присваивается значение -Inf.

        Args:
            logits: Исходные логиты.
            top_k: Количество претендентов для top-k сэмплирования.
            top_p: Вероятностная масса для top-p сэмплирования.

        Returns:
            Преобразованные логиты.
        """
        if top_k:
            mask = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(mask, -float('Inf'))
        if top_p:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            sorted_probs_cumsum = torch.cumsum(sorted_probs, dim=-1)
            
            mask = sorted_probs_cumsum >= top_p
            mask[:, 0] = 0
            mask = mask.scatter(-1, sorted_indices, mask)
            logits = logits.masked_fill(mask, -float('Inf'))
            
        return logits
        
    def fit(self,
            train_loader,
            valid_loader,
            num_epoch: int,
            learning_rate: float):
        """Определяет логику обучения модели.

        Args:
            train_loader: torch.utils.data.DataLoader, содержит данные для обучения.
            valid_loader: torch.utils.data.DataLoader, содержит данные для валидации.
            num_epoch: Количество эпох обучения.
            learning_rate: Скорость обучения.
        """
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for i in range(num_epoch):

            train_losses = []
            self.train()
            for inputs, targets in train_loader:
                logits, _ = self.forward(inputs, use_cache=False)
                logits = logits.reshape(-1, self.vocab_size)
                targets = targets.flatten()

                loss = criterion(logits, targets)
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {i} mean train loss: {torch.mean(torch.FloatTensor(train_losses))}')

            valid_losses = []
            self.eval()
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    logits, _ = self.forward(inputs, use_cache=False)
                    logits = logits.reshape(-1, self.vocab_size)
                    targets = targets.flatten()

                    loss = criterion(logits, targets)
                    valid_losses.append(loss.item())
            print(f'Epoch {i} mean valid loss: {torch.mean(torch.FloatTensor(valid_losses))}')
            
            self.save(path=f'current_model_epoch_{i}.pth')

        return

    def save(self, path: str):
        """Сохранение обученной модели.

        Args:
            path: Путь для сохранения.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)
        
    @classmethod
    def load(cls,
             path: str,
             device: str):
        """Загрузка обученной модели.

        Args:
            path: Путь для загрузки.
            device: Где хранить и совершать вычисления.

        Returns:
            Загруженная модель.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model