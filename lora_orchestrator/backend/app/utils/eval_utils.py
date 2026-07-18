import torch
import numpy as np
from typing import List
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    smooth = SmoothingFunction().method1
    refs = [[r.split()] for r in references]
    hyps = [p.split() for p in predictions]
    return corpus_bleu(refs, hyps, smoothing_function=smooth)


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    return float(np.mean(scores))


def compute_perplexity(model, tokenizer, texts: List[str], max_length: int = 512) -> float:
    if not texts:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    if total_tokens == 0:
        return 0.0
    return float(np.exp(total_loss / total_tokens))


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions)


def run_inference(model, tokenizer, prompts: List[str], batch_size: int = 4) -> List[str]:
    model.eval()
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(decoded)
    return all_outputs
