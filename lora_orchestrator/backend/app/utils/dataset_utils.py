import json
import os
from typing import Tuple, List
import pandas as pd
import numpy as np


def load_dataset(path: str, fmt: str, max_samples: int) -> pd.DataFrame:
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                records.append(json.loads(line.strip()))
        df = pd.DataFrame(records)
    elif fmt == "huggingface":
        from datasets import load_dataset as hf_load
        ds = hf_load(path, split="train")
        df = pd.DataFrame(ds.to_list()[:max_samples])
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if len(df) > max_samples:
        df = df.head(max_samples)
    return df


def deduplicate(df: pd.DataFrame, col: str = "instruction") -> int:
    before = len(df)
    df.drop_duplicates(subset=[col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return before - len(df)


def filter_empty(df: pd.DataFrame, min_chars: int = 10) -> int:
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in ["instruction", "output"]:
        if col in df.columns:
            mask &= df[col].astype(str).str.len() >= min_chars
    df.drop(df[~mask].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return before - len(df)


def format_alpaca(row: dict) -> dict:
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", "")
    input_section = ""
    if inp and str(inp).strip():
        input_section = ", paired with an input that provides further context"
    text = (
        f"Below is an instruction that describes a task{input_section}. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )
    return {"text": text}


def format_sharegpt(row: dict) -> dict:
    conversations = row.get("conversations", [])
    return {"conversations": conversations}


def compute_token_stats(texts: List[str], tokenizer) -> dict:
    token_lengths = [len(tokenizer.encode(t)) for t in texts]
    return {
        "avg_token_length": float(np.mean(token_lengths)),
        "max_token_length": int(np.max(token_lengths)),
        "min_token_length": int(np.min(token_lengths)),
        "p95_token_length": int(np.percentile(token_lengths, 95)),
    }


def split_dataset(df: pd.DataFrame, holdout_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    holdout_size = max(1, int(len(df) * holdout_ratio))
    holdout = df.sample(n=holdout_size, random_state=42)
    train = df.drop(holdout.index)
    return train, holdout


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
