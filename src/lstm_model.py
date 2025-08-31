import os, re
from typing import List
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from common import top_p_filtering

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_GEN_LEN = 20

class LSTMWordGenerator(nn.Module):
    """
    Train: Embedding → (Bi)LSTM → Linear (vocab size)
    Inference: Embedding → LSTM → Linear (vocab size)
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        lstm_out_dim = hidden_dim
        self.fc = nn.Linear(lstm_out_dim, vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None):
        '''
        Возвращает логиты и ошибку
        '''
        embeds = self.embedding(input_ids)               # (B, L, D)

        # защита от нулевых длин
        lengths = attention_mask.sum(dim=1).cpu()        # (B,)
        # Если в батче есть полностью пустые примеры (length == 0),
        # заменяем 0 на 1, чтобы pack_padded_sequence не падала.
        if (lengths == 0).any():
            lengths = lengths.clone()
            lengths[lengths == 0] = 1

        packed = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        logits = self.fc(out)

        if labels is None:
            return {'logits': logits}
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {'logits': logits, 'loss': loss}

    @torch.no_grad()
    def generate_one_word(self, 
                            text_prompt: str, 
                            tokenizer: AutoTokenizer, 
                            eos_id: int | None = None) -> str:
        """
        Генерирует одно слово после заданного текстового префикса
        """
        device = next(self.parameters()).device
        self.eval()

        # Токенизируем текст и получаем IDs
        inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)

        # Выполняем прямой проход через модель
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][:, -1, :]  # Берем последние токены

        # Предсказываем следующий токен
        next_token = torch.argmax(logits, dim=-1)[0].item()

        # Преобразуем токен обратно в текст
        word = tokenizer.decode([next_token])
        return word

    @torch.no_grad()
    def generate_n_words(self, 
                            text_prompt: str, 
                            n: int, 
                            tokenizer: AutoTokenizer, 
                            eos_id: int | None = None, 
                            do_sampling: bool = False, 
                            temperature: float = 1.0, 
                            top_p: float = 0.9) -> str:
        """
        Генерирует N новых слов после заданного текстового префикса.
        """
        device = next(self.parameters()).device
        self.eval()

        # Токенизируем входной текст
        inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)

        # Начальные токены
        current_ids = input_ids

        with torch.no_grad():
            for _ in range(n):
                # Создаем внимание на всю длину текущих токенов
                att_mask = torch.ones_like(current_ids, device=device)

                # Прогоняем через модель
                outputs = self.forward(input_ids=current_ids, attention_mask=att_mask)
                logits = outputs["logits"][:, -1, :]  # Последние токены

                if do_sampling:
                    # Применяем температуру
                    logits /= max(temperature, 1e-8)

                    # Отсекаем редкие токены по Top-P
                    filtered_probs = F.softmax(top_p_filtering(logits.squeeze(), top_p=top_p), dim=-1)
                    next_token = torch.multinomial(filtered_probs, num_samples=1).unsqueeze(0)
                else:
                    # Просто выбираем самый вероятный токен
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Добавляем токен в последовательность
                current_ids = torch.cat((current_ids, next_token), dim=1)

                # Проверка на достижение символа конца строки
                if eos_id is not None and next_token.item() == eos_id:
                    break

        # Переводим токены обратно в текст
        result_text = tokenizer.decode(current_ids.squeeze().tolist())
        return result_text[len(text_prompt):].strip()  # Убираем оригинальный текст и лишние пробелы


    def generate_one_sample(self,
                            prompt_ids: List[int],
                            eos_id: int) -> List[int]:
        """
        Жадная генерация (по-умолчанию).  
        Останавливается, когда сгенерирован `eos_id` или
        достигнут `max_gen_len`.

        prompt_ids - Список токенов-промпта (может уже содержать `eos_id`).
        eos_id - ID токена конца предложения.
        gen_len - Максимальное число токенов, которое будет добавлено
        к уже существующему промпту.

        Возвращает Полный список токенов (промпт+сгенерированное продолжение).
        """
        device = next(self.parameters()).device
        self.eval()

        # Защита от пустого промпта (аналогично generate)
        if not prompt_ids:
            return [] if eos_id is None else [eos_id]

        # Приводим промпт к тензору формы (1, L)
        generated = torch.tensor(prompt_ids,
                                 dtype=torch.long,
                                 device=device).unsqueeze(0)   # (1, L)

        with torch.no_grad():
            for _ in range(MAX_GEN_LEN):
                # учитывает все уже сгенерированные токены
                attn_mask = torch.ones_like(generated, device=device)

                out = self(input_ids=generated,
                           attention_mask=attn_mask)          # logits: (1, cur_len, vocab)
                next_token_logits = out['logits'][:, -1, :]      # (1, vocab)

                # Жадный выбор (можно заменить на sampling/beam-search)
                next_token = torch.argmax(next_token_logits, dim=-1)  # (1)

                # Добавляем выбранный токен к последовательности
                generated = torch.cat([generated,
                                      next_token.unsqueeze(-1)], dim=1)

                # Прерываем, если получили EOS
                if eos_id is not None and next_token.item() == eos_id:
                    break

        # Возвращаем чистый список int-ов (без batch-измерения)
        return generated.squeeze().tolist()        
    
    def tokens_to_words(self, 
                        gen_ids: List[int],
                        eos_id: int,
                        tokenizer: AutoTokenizer) -> List[str]:
        '''
        Получает сгенерированные айдишнки
        Берет полезную часть (до паддинга)
        На выходе декодированный текст
        '''
        # Обрезаем сгенерированный токен-список до EOS (если он есть)
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]

        # Декодируем оба текста
        gen_words = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_words
    
    def words_to_tokens(
        self,
        tweet: str,
        tokenizer: AutoTokenizer,
        add_special_tokens: bool = False) -> List[str]:
        """
        Преобразует один твит в список токенов.
        """
        enc = tokenizer(tweet,
                         truncation=True,
                         max_length=MAX_GEN_LEN,
                         add_special_tokens=add_special_tokens,
                         padding=False)
        prompt_ids = [ids for ids in enc['input_ids']]
        return prompt_ids

    def complete(self, *, 
                 text: str,
                 eos_id: int,
                 tokenizer: AutoTokenizer,
                 add_special_tokens=False,
                 preprocess: bool = True) -> str:
        
        prompt_ids = self.words_to_tokens(text, tokenizer, add_special_tokens, MAX_GEN_LEN)
        generated_ids = self.generate(prompt_ids=prompt_ids,
                                      eos_id=eos_id,
                                      do_sampling=True,
                                      temperature=0.7,
                                      top_p=0.8)
        generated_text = self.tokens_to_words(generated_ids, eos_id, tokenizer)
        return generated_text

    def generate(self,
                prompt_ids: List[int],
                eos_id: int,
                do_sampling: bool = False,
                temperature: float = 1.0,
                top_p: float = 0.9) -> List[int]:
        """
        Универсальный генератор, поддерживает:
        * greedy (do_sampling=False)
        * sampling с temperature / top-p (do_sampling=True)
        """
        device = next(self.parameters()).device
        self.eval()

        # Защита от полностью пустого промпта
        if not prompt_ids:                     # ничего не генерируем
            # Возвращаем либо пустой список, либо [eos_id] – выбираем
            # вариант, который проще обрабатывается дальше.
            return [] if eos_id is None else [eos_id]

        # (1, L) – уже tokenы промпта
        generated = torch.tensor(prompt_ids,
                                 dtype=torch.long,
                                 device=device).unsqueeze(0)   # (1, L)

        with torch.no_grad():
            for _ in range(MAX_GEN_LEN):
                attn_mask = torch.ones_like(generated, device=device)

                out = self(input_ids=generated,
                           attention_mask=attn_mask)  # logits (1, cur_len, vocab)
                next_logits = out["logits"][:, -1, :]                     # (1, vocab)

                if do_sampling:
                    # temperature
                    logits = next_logits / max(temperature, 1e-8)

                    # top-p
                    logits = top_p_filtering(logits.squeeze(0), top_p=top_p).unsqueeze(0)

                    # выбор из распределения
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)   # (1,1)
                else:            # greedy
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1,1)

                generated = torch.cat([generated, next_token], dim=1)

                # EOS?
                if eos_id is not None and next_token.item() == eos_id:
                    break

        return generated.squeeze().tolist()