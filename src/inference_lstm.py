import os, re
import torch

from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from data_utils import make_val_loader
from train_lstm import evaluate_on_loader


os.environ["TOKENIZERS_PARALLELISM"] = "false"
CSV_PATH = "./data/training.1600000.processed.noemoticon.csv"
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
URL_RE      = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
MENTION_RE  = re.compile(r'@\w+')
HASHTAG_RE  = re.compile(r'#(\w+)')
NUM_WORKERS = 0
COMPARISON_DS_SIZE = 100 # количество записей из датасета для сравнения трансформера и lstm
MAX_GEN_LEN = 20


def inference_lstm(*,
                   tokenizer: AutoTokenizer,
                   eos_id: int,
                   use_sampling: bool = False,
                   temperature: float = 1.0,
                   top_p: float = 0.9):
    
    device = torch.device("cpu")
    best_lstm_path = "./models/best_models/full_final_model.pt"
    model = torch.load(best_lstm_path, map_location=device)
    model.eval()

    # Формируем DataLoader для validation-части.
    val_loader, hf, val_texts = make_val_loader(tokenizer=tokenizer,
                                                    eos_id=eos_id,
                                                    device=device,
                                                    nrows=COMPARISON_DS_SIZE)

    # Вычисляем метрики. verbose=True заставит функцию вернуть
    #      список ROUGE-L по каждому примеру.
    ppl, avg_rouge, per_example = evaluate_on_loader(model=model,
                                                        loader=val_loader,
                                                        tokenizer=tokenizer,
                                                        device=device,
                                                        eos_id=eos_id,
                                                        fraction=0.5,
                                                        verbose=True,
                                                        use_sampling=use_sampling,
                                                        temperature=temperature,
                                                        top_p=top_p)

    val_hf = hf
    autocompleted_texts = []
    for i in range(len(val_hf)):
        ids = val_hf[i]["input_ids"]
        # убираем PAD (0) и EOS, если они есть
        if 0 in ids:
            ids = ids[:ids.index(0)]
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        # получаем чистый текст твита
        src_text = tokenizer.decode(ids, skip_special_tokens=True)
        # генерируем продолжение (используем уже обученную модель)
        fraction = 0.5
        split_idx = int(len(src_text) * fraction)
        prompt_text = src_text[:split_idx] 
        compl = model.generate(
            prompt_ids=tokenizer.encode(prompt_text, add_special_tokens=False),
            eos_id=eos_id,
            do_sampling=use_sampling,
            temperature=temperature,
            top_p=top_p,
        )
        compl_text = tokenizer.decode(compl, skip_special_tokens=True)
        autocompleted_texts.append({'prompt': prompt_text, 
                                    'prediction': compl_text})

    return avg_rouge, autocompleted_texts