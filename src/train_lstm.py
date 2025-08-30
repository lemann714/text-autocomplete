import os, math, itertools, re
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import (
    DatasetDict,
    concatenate_datasets,          
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from data_utils import token_collate_fn, TweetDataset, split_hf_dataset, read_dataset
from lstm_model import LSTMWordGenerator
from common import rouge_l_f1, set_seed

from inference_transformer import TOKENIZER, EOS_ID


os.environ["TOKENIZERS_PARALLELISM"] = "false"
CSV_PATH = "./data/training.1600000.processed.noemoticon.csv"
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
URL_RE      = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
MENTION_RE  = re.compile(r'@\w+')
HASHTAG_RE  = re.compile(r'#(\w+)')
NUM_WORKERS = 0
COMPARISON_DS_SIZE = 100 # количество записей из датасета для сравнения трансформера и lstm
MAX_GEN_LEN = 20


def complete_text(
    prompt_ids: List[int],
    model: AutoModelForCausalLM,
    eos_id: int,
    do_sampling: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> List[int]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_kwargs = {
        "max_new_tokens": MAX_GEN_LEN,
        "eos_token_id": eos_id,
        "do_sample": do_sampling,
    }
    if do_sampling:
        gen_kwargs.update(
            {"temperature": temperature, "top_p": top_p}
        )
    out = model.generate(input_ids, **gen_kwargs)          # (1, prompt+generated)
    generated = out[0][input_ids.shape[-1] :].tolist()    # только сгенерированное
    return generated

def evaluate_on_loader(model: nn.Module,
                        loader: DataLoader,
                        tokenizer: AutoTokenizer,
                        device: torch.device,
                        eos_id: int,
                        fraction: float = 0.5,
                        use_sampling: bool = False,
                        temperature: float = 0.7,
                        top_p: float = 0.9,
                        verbose: bool = False) -> Tuple[float, float, List[float] | None]:
    """
    Возвращает (perplexity, avg_rougeL, per_example_scores|None)
    Оценка одинаково работает для LSTM и для трансформера,
    при условии, что модель реализует метод `generate_one_sample`
    (у LSTM он уже есть, у трансформера реализуем «обёртку» ниже).
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    rouge_sum, n_examples = 0.0, 0
    per_example = [] if verbose else None

    with torch.no_grad():
        for batch in loader:
            # perplexity
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=labels)
            loss = out["loss"]
            # количество токенов без паддинга:
            n_tok = attn_mask.sum().item()
            total_nll += loss.item() * n_tok
            total_tokens += n_tok

            #  ROUGE 
            ids_np = input_ids.cpu().numpy()
            for i in range(ids_np.shape[0]):
                seq = ids_np[i].tolist()
                # убираем PAD и EOS
                if 0 in seq:
                    seq = seq[:seq.index(0)]
                if eos_id in seq:
                    seq = seq[:seq.index(eos_id)]

                if not seq:
                    continue

                split = int(len(seq) * fraction)
                prompt_ids = seq[:split]
                ref_ids    = seq[split:]

                # генерация (можно использовать единый интерфейс)
                if isinstance(model, LSTMWordGenerator):
                    gen_ids = model.generate(prompt_ids=prompt_ids,
                                            eos_id=eos_id,
                                            do_sampling=use_sampling,
                                            temperature=temperature,
                                            top_p=top_p)
                else:   # трансформер
                    gen_ids = complete_text(prompt_ids=prompt_ids,
                                            tokenizer=tokenizer,
                                            model=model,
                                            eos_id=eos_id,
                                            do_sampling=use_sampling,
                                            temperature=temperature,
                                            top_p=top_p)
                # декодируем
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

                rouge_f = rouge_l_f1(ref_text, gen_text)
                rouge_sum += rouge_f
                n_examples += 1
                if verbose:
                    per_example.append(rouge_f)

    perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    avg_rouge = rouge_sum / n_examples if n_examples > 0 else 0.0
    return perplexity, avg_rouge, per_example

def grid_search(
    param_grid: Dict[str, List[Any]],
    base_train_loader: DataLoader,
    base_val_loader: DataLoader,
    base_test_loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    eos_id: int,
    results_dir: str = "models") -> pd.DataFrame:
    """
    Перебирает все комбинации параметров, обучает модель,
    сохраняет лучшую модель и графики, возвращает таблицу-результатов.
    """
    print('\nИщем оптимальные параметры обучения\n')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "best_models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    # список всех вариантов в виде dict
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_records = []

    for idx, cfg in enumerate(combos, start=1):
        print(f"\nРассмотри {idx}/{len(combos)} – cfg: {cfg}\n")

        lr          = cfg["lr"]
        embed_dim   = cfg["embed_dim"]
        hidden_dim  = cfg["hidden_dim"]
        num_layers  = cfg["num_layers"]
        bidirectional = cfg["bidirectional"]
        batch_size  = cfg["batch_size"]
        epochs      = cfg["epochs"]

        # пересоздаём даталоадеры, если меняется batch_size
        train_loader = DataLoader(
            base_train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=token_collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            base_val_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=token_collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            base_test_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=token_collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

        vocab_size = tokenizer.vocab_size
        model = LSTMWordGenerator(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
        ).to(device)

        best_path = os.path.join(
            results_dir, "best_models", f"run_{idx:03d}_best.pt"
        )
        model, history = train(
            model,
            train_loader,
            val_loader,
            tokenizer,
            device,
            eos_id,
            epochs=epochs,
            lr=lr,
            warmup_ratio=0.0,         
            patience=3,
            best_model_path=best_path,
        )

        test_ppl, test_rouge, _ = evaluate_on_loader(
            model,
            test_loader,
            tokenizer,
            device,
            eos_id,
            fraction=0.5,
        )
        print(f"Test PPL: {test_ppl:.2f} |  Test ROUGE-L: {test_rouge:.4f}")

       # train_loss + val_rougeL
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(
            [h["epoch"] for h in history],
            [h["train_loss"] for h in history],
            label="train loss",
            marker="o",
            color="#1f77b4",
        )
        ax1.plot(
            [h["epoch"] for h in history],
            [h["val_rougeL"] for h in history],
            label="val ROUGE-L",
            marker="s",
            color="#ff7f0e",
        )
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("value")
        ax1.set_title(f"Run {idx:03d} – train loss / val ROUGE-L")
        ax1.legend()
        plt.tight_layout()
        fig1.savefig(
            os.path.join(results_dir, "plots", f"run_{idx:03d}_loss_rouge.png")
        )
        plt.close(fig1)

        # train_loss + val_perplexity
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(
            [h["epoch"] for h in history],
            [h["train_loss"] for h in history],
            label="train loss",
            marker="o",
            color="#1f77b4",
        )
        ax2.plot(
            [h["epoch"] for h in history],
            [h["val_perplexity"] for h in history],
            label="val perplexity",
            marker="x",
            color="#2ca02c",
        )
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("value")
        ax2.set_title(f"Run {idx:03d} – train loss / val perplexity")
        ax2.legend()
        plt.tight_layout()
        fig2.savefig(
            os.path.join(results_dir, "plots", f"run_{idx:03d}_loss_ppl.png")
        )
        plt.close(fig2)

        # аписываем строку в итоговую таблицу
        record = {
            **cfg,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_perplexity": history[-1]["val_perplexity"],
            "final_val_rougeL": history[-1]["val_rougeL"],
            "test_perplexity": test_ppl,
            "test_rougeL": test_rouge,
            "best_model_path": best_path,
        }
        all_records.append(record)

        # сохраняем промежуточный CSV после каждой итерации (чтобы не потерять результаты)
        pd.DataFrame(all_records).to_csv(
            os.path.join(results_dir, "grid_search_results.csv"),
            index=False,
        )

    #возврат полной таблицы
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(
        os.path.join(results_dir, "grid_search_results.csv"),
        index=False,
    )
    return results_df


def train(model: LSTMWordGenerator,
          train_loader: DataLoader,
          val_loader: DataLoader,
          tokenizer: AutoTokenizer,
          device: torch.device,
          eos_id: int,
          epochs: int = 10,
          lr: float = 5e-4,
          warmup_ratio: float = 0.1,
          patience: int = 2,
          best_model_path: str = 'best_blstm.pt') -> LSTMWordGenerator:
    '''
    Собирает историю метрик (train_loss, val_ppl, val_rouge) для каждой эпохи.

    Возвращает обученную модель (с лучшими весами) и список словарей-записей.
    '''
    total_steps = len(train_loader) * epochs
    # warmup_steps = int(warmup_ratio * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_training_steps=total_steps,
    #     num_warmup_steps=warmup_steps
    # )

    best_val_rouge = -float('inf')
    no_improve = 0
    history = []

    # train
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device))
            
            loss = out['loss']
            loss.backward()
            optimizer.step()
            # scheduler.step()

            epoch_loss += loss.item() * batch['input_ids'].size(0)
        avg_train_loss = epoch_loss / len(train_loader)

        # validation
        val_ppl, val_rouge, _ = evaluate_on_loader(model,
                                                val_loader,
                                                tokenizer,
                                                device,
                                                eos_id)
        print(f'\nEpoch {epoch:02d} | train_loss={avg_train_loss:.4f}'
              f' | valid. ppl={val_ppl:.2f} | valid.ROUGE-L={val_rouge:.4f}')
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_perplexity": val_ppl,
                "val_rougeL": val_rouge,
            }
        )
        # early stopping
        if val_rouge > best_val_rouge:
            best_val_rouge = val_rouge
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # загрузим лучшую
    model.load_state_dict(torch.load(best_model_path))
    return model, history


def train_final(best_cfg: dict,
                splits: DatasetDict,
                tokenizer: AutoTokenizer,
                device: torch.device,
                eos_id: int,
                final_model_path: str = 'full_final_model.pt',
                results_dir: str = 'models'):
    '''
    Объединяет train, val, test. обучает модель с лучшими гиперпараметрами
    и сохраняет её в <results_dir>/final_model.pt.
    Возвращает обученную модель.
    '''
    print('\nОбучение финальной модели...\n')
    full_dataset = concatenate_datasets([splits["train"],
                                         splits["validation"],
                                         splits["test"]])

    full_tweet_ds = TweetDataset(full_dataset, eos_id=eos_id)
    batch_size = best_cfg["batch_size"]
    # Даталоадер (только train-loader, валидации нет)
    train_loader = DataLoader(
        full_tweet_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=token_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    # Инициализируем модель с найденными гиперпараметрами
    vocab_size = tokenizer.vocab_size
    model = LSTMWordGenerator(
        vocab_size=vocab_size,
        embed_dim=best_cfg["embed_dim"],
        hidden_dim=best_cfg["hidden_dim"],
        num_layers=best_cfg["num_layers"],
        bidirectional=best_cfg["bidirectional"]).to(device)
    
    # обучаем с использованием early-stopping для защиты от переобучения
    # но теперь мониторим ROUGE-L на 10% от полной выборки
    split_tmp = full_dataset.train_test_split(test_size=0.10, seed=42)
    val_hf = split_tmp["test"]
    val_ds = TweetDataset(val_hf, eos_id=eos_id)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=token_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    epochs = best_cfg["epochs"] * 2 # удваиваем чтобы дать модели шанс разойтись на большом объёме, но early-stop всё равно прервет
    lr = best_cfg["lr"]
    patience = 3

    model, _ = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        eos_id=eos_id,
        epochs=epochs,
        lr=lr,
        warmup_ratio=0.0,         
        patience=patience,
        best_model_path=os.path.join(results_dir, "best_models/final_model.pt")
    )

    torch.save(model, os.path.join(results_dir, f"best_models/{final_model_path}"))

    print(f"\nВеса итоговой модели сохранены в {os.path.join(results_dir, 'final_model.pt')}")
    print(f"\nПолная финальная модель сохранена в {os.path.join(results_dir, 'full_final_model.pt')}")
    return model


def train_lstm(tokenizer: AutoTokenizer,
               eos_id: int):
    # Поиск оптимальных параметров обучения
    grid = {
        "lr":          [5e-4],
        "embed_dim":   [128],
        "hidden_dim":  [128],
        "num_layers":  [1],
        "bidirectional": [False],
        "batch_size":  [16],            
        "epochs":      [4],
    }

    # общие настройки ----------
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Чтение и токенизация ----------
    # MAX_LEN  = 128
    NROWS    = None if device.type == "cuda" else 50              
    hf, cleaned_texts = read_dataset(CSV_PATH, 
                                     tokenizer, 
                                     max_length=tokenizer.model_max_length, 
                                     nrows=NROWS)

    # делим на сплиты ----------
    splits = split_hf_dataset(
        hf, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=2025
    )

    # даталоадеры будут пересозданы внутри grid_search
    # они нужны только чтобы взять `dataset` и параметры `num_workers/pin_memory`
    base_train_ds = TweetDataset(splits["train"], eos_id=eos_id) 
    base_val_ds   = TweetDataset(splits["validation"], eos_id=eos_id)
    base_test_ds  = TweetDataset(splits["test"], eos_id=eos_id)
    base_train_loader = DataLoader(
        base_train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=token_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    base_val_loader = DataLoader(
        base_val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=token_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    base_test_loader = DataLoader(
        base_test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=token_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    # Запуск grid-search
    results_df = grid_search(
        param_grid=grid,
        base_train_loader=base_train_loader,
        base_val_loader=base_val_loader,
        base_test_loader=base_test_loader,
        tokenizer=tokenizer,
        device=device,
        eos_id=eos_id,
        results_dir="models",
    )

    # Находим лучшую конфигурацию (по тестовому ROUGE-L)
    best_cfg = results_df.loc[results_df["test_rougeL"].idxmax()].to_dict()
    print("\nЛучшая модель (по test-rougeL)")
    print(best_cfg)
    
    # Обучение финальной модели (без сплита данных)
    final_model = train_final(best_cfg=best_cfg,
                              splits=splits,
                              tokenizer=tokenizer,
                              device=device,
                              eos_id=eos_id,
                              results_dir='models')
    return final_model


if __name__=="__main__":
    # обучаем именно на этом токенизаторе
    train_lstm(tokenizer=TOKENIZER,
                eos_id=EOS_ID)