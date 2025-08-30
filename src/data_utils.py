from transformers import AutoTokenizer
from datasets import (
    DatasetDict,
    Dataset as HFDataset,
)
import pandas as pd
import re, unicodedata
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence


CSV_PATH = "../data/training.1600000.processed.noemoticon.csv"
URL_RE      = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
MENTION_RE  = re.compile(r'@\w+')
HASHTAG_RE  = re.compile(r'#(\w+)')
NUM_WORKERS = 0


def read_dataset(csv_path: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 128,
                 nrows: int | None = None) -> HFDataset:
    """
    Возвращает HuggingFace-Dataset с колонками:
        - input_ids (list[int])
        - attention_mask (list[int])
    """
    col_names = ["target", "ids", "date", "flag", "user", "text"]
    df_raw = pd.read_csv(
        csv_path,
        header=None,
        names=col_names,
        encoding="latin1",
        dtype=str,
        usecols=["text"],
        na_values=["NO_QUERY"],
        keep_default_na=False,
        on_bad_lines="skip",
        nrows=nrows,
    )
    df_raw["text"] = df_raw["text"].apply(clean_tweet)
    cleaned_texts = df_raw["text"].tolist()
    dataset = HFDataset.from_pandas(df_raw)
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"],
                               truncation=True,
                               max_length=max_length,
                               padding=False,
                               return_attention_mask=True),
        batched=True,
        remove_columns=["text"]
    )
    return tokenized, cleaned_texts

def split_hf_dataset(hf_dataset: HFDataset,
                     train_ratio: float = 0.80,
                     val_ratio:   float = 0.10,
                     test_ratio:  float = 0.10,
                     seed: int = 42) -> DatasetDict:
    '''
    Принимает токенизированный Dataset и возвращает
    три датасета:
        - train
        - validation
        - test

    Параметры:
        train_ratio, val_ratio, test_ratio – доли.
        seed – фиксирует случайность, чтобы результаты были воспроизводимы.
    '''
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Все доли должны лежать в (0,1)")
    split1 = hf_dataset.train_test_split(test_size=1.0 - train_ratio, seed=seed)
    train_ds = split1['train']
    rest = split1['test']

    val_rel = val_ratio / (val_ratio + test_ratio)
    split2 = rest.train_test_split(test_size=1.0 - val_rel, seed=seed)

    val_ds = split2['train']
    test_ds = split2['test']

    return DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })


def remove_emoji(text: str) -> str:
    """
    Удаляет все символы, у которых Unicode-категория 
    начинается с 'S' (Symbol) или 'C' (Other, в т.ч. 
    контрольные символы).
    """
    return ''.join(ch for ch in text
                   if not (unicodedata.category(ch).startswith('S') or
                           unicodedata.category(ch).startswith('C')))

def clean_tweet(text_as_is: str) -> str:
    """
    text_as_is - твит
    
    Убирает:
        ссылки
        упоминания
        эмодзи
        лишние пробелы

    Возвращает: очищенный твит
    """
    if not isinstance(text_as_is, str):
        return ''

    text = text_as_is.lower()
    text = URL_RE.sub('', text)
    text = MENTION_RE.sub('', text)
    text = HASHTAG_RE.sub(r'\1', text)
    text = remove_emoji(text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def make_val_loader(
    tokenizer: AutoTokenizer,
    eos_id: int,
    device: torch.device,
    nrows: int | None = None,
) -> Tuple[DataLoader, DatasetDict, List[str]]:
    
    VAL_BATCH_SIZE = 10 # размер батча для сравнения трансформера и lstm
    """
    Читает датасет, делит его на splits и возвращает
    val_loader – DataLoader, готовый к использованию LSTM-моделью;
    splits – словарь с HF-сплитами (можно взять validation-датасет);
    val_texts – список чистых твитов (строк), которые нужны трансформеру.
    """
    hf_tokenized, cleaned_texts = read_dataset(CSV_PATH,
                                               tokenizer, 
                                               max_length=tokenizer.model_max_length,
                                               nrows=nrows)
    val_ds = TweetDataset(hf_tokenized, eos_id=eos_id)
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=token_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_indices = hf_tokenized["__index__"] if "__index__" in hf_tokenized.features else None
    if val_indices is not None:
        # Если у HF-датасета есть поле __index__, используем его.
        val_texts = [cleaned_texts[i] for i in val_indices]
    else:
        # Если индексов нет (в старых версиях), просто берём первые N
        # элементов, где N = len(splits["validation"]).
        val_texts = cleaned_texts[:len(hf_tokenized)]
    return val_loader, hf_tokenized, val_texts, 

class TweetDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset, eos_id: int):
        self.input_ids = [ex["input_ids"] for ex in hf_dataset]
        self.attn_mask = [ex["attention_mask"] for ex in hf_dataset]
        self.eos_id = eos_id

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (torch.tensor(self.input_ids[idx], dtype=torch.long),
                torch.tensor(self.attn_mask[idx], dtype=torch.long))
    

def token_collate_fn(batch):
    # batch = [(ids, mask), …]
    ids, masks = zip(*batch)
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    # Для LSTM нам нужны также `labels` – сдвинутые на 1 токен
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100          # ignore last token
    return {"input_ids": ids,
            "attention_mask": masks,
            "labels": labels}