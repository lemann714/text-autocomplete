import os, re

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import make_val_loader
from common import rouge_l_f1, get_eos_id

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CSV_PATH = "./data/training.1600000.processed.noemoticon.csv"
URL_RE      = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
MENTION_RE  = re.compile(r'@\w+')
HASHTAG_RE  = re.compile(r'#(\w+)')
NUM_WORKERS = 0
COMPARISON_DS_SIZE = 100 # количество записей из датасета для сравнения трансформера и lstm
MAX_GEN_LEN = 20
TRANSFORMER_NAME = "distilgpt2"

TOKENIZER = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
EOS_ID = get_eos_id(TOKENIZER)

def generate_completion(prompt: str,
                        tokenizer: AutoTokenizer,
                        model: AutoModelForCausalLM,
                        temperature: float = 1.0,
                        top_p: float = 0.9,
                        do_sample: bool = False) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    generated = out_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def validate_on_dataset(texts: list[str],
                        tokenizer: AutoTokenizer,
                        model: AutoModelForCausalLM,
                        fraction: float = 0.5,
                        do_sampling: bool = False,
                        temperature: float = 1.0,
                        top_p: float = 0.9) -> dict:
    """
    Для каждого текста берём первые 75 % как prompt,
    а оставшиеся – как reference.
    """
    f1_scores = []
    examples = []

    for txt in texts:
        split_idx = int(len(txt) * fraction)
        prompt, reference = txt[:split_idx], txt[split_idx:]

        pred = generate_completion(
            prompt,
            tokenizer=tokenizer,
            model=model,
            temperature=temperature,
            top_p=top_p,
            # switch between greedy / sampling
            do_sample=do_sampling
        )

        score = rouge_l_f1(reference, pred)
        f1_scores.append(score)

        examples.append({
            "prompt":      prompt,
            "prediction":  pred
        })

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"rougeL_f1": avg_f1, "examples": examples}

def inference_transformer(*,
                          transformer_name: str,
                          tokenizer: AutoTokenizer,
                          eos_id: int,
                          use_sampling: bool = False,
                          temperature: float = 1.0,
                          top_p: float = 0.9):
    model = AutoModelForCausalLM.from_pretrained(transformer_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.to(device)

    # Получаем готовый DataLoader и **тексты** для трансформера
    val_loader, hf, val_texts = make_val_loader(
        tokenizer=tokenizer,          
        eos_id=eos_id,
        device=device,
        nrows=COMPARISON_DS_SIZE,                     
    )

    # Валидация трансформера (ROUGE-L F1)
    results = validate_on_dataset(
                texts=val_texts,
                tokenizer=tokenizer,
                model=model,
                fraction=0.5,
                # передаём параметры в generate()
                do_sampling=use_sampling,
                temperature=temperature,
                top_p=top_p,
    )
    avg_rouge = results['rougeL_f1']
    autocompleted_texts = results["examples"] 
    return avg_rouge, autocompleted_texts

