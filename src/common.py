from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import Tensor
import torch
from rouge_score import rouge_scorer
import random
import numpy as np
import torch.nn.functional as F


ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def get_eos_id(tokenizer: AutoTokenizer) -> int:
    """Возвращает id конца последовательности"""
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.sep_token_id is not None:
        return tokenizer.sep_token_id
    raise ValueError("Tokenizer has neither eos_token nor sep_token.")

def rouge_l_f1(ref: str, hyp: str) -> float:
    return ROUGE.score(ref, hyp)['rougeL'].fmeasure

def top_p_filtering(logits: Tensor, top_p: float = 0.9) -> Tensor:
    """
    Оставляем только те токены, чья суммарная вероятность (по убыванию) ≤ top_p.
    Возвращаем logits, где остальные токены заменены на -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Оставляем токены, пока кумулятивная вероятность ≤ top_p
    sorted_indices_to_keep = cumulative_probs <= top_p
    # гарантируем, что как минимум один токен остаётся (первый)
    sorted_indices_to_keep[..., 0] = 1

    # Маска: -inf для токенов, которые выкинули
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(~sorted_indices_to_keep, float("-inf")),
    )
    return mask

def set_seed(seed: int = 42):
    '''
    Позволяет выполнять детерминированный запуск
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False