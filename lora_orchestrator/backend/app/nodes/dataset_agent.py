import os
import json
from datetime import datetime, timezone
from backend.app.state import DatasetStats
from backend.app.utils import dataset_utils


def dataset_node(state: dict, llm=None) -> dict:
    cfg = state["run_config"]
    session_id = cfg["session_id"]
    
    # Load
    df = dataset_utils.load_dataset(cfg["dataset_path"], cfg["dataset_format"], cfg["max_samples"])
    
    # Deduplicate
    dups_removed = dataset_utils.deduplicate(df)
    
    # Filter empty
    empty_removed = dataset_utils.filter_empty(df)
    
    # Format
    formatted_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if cfg["template_format"] == "alpaca":
            formatted_rows.append(dataset_utils.format_alpaca(row_dict))
        else:
            formatted_rows.append(dataset_utils.format_sharegpt(row_dict))
    
    # Token stats
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_id"], use_fast=True, trust_remote_code=True)
    texts = [r.get("text", "") for r in formatted_rows if "text" in r]
    if not texts:
        texts = [json.dumps(r) for r in formatted_rows]
    token_stats = dataset_utils.compute_token_stats(texts, tokenizer)
    
    # Split
    import pandas as pd
    fmt_df = pd.DataFrame(formatted_rows)
    train_df, holdout_df = dataset_utils.split_dataset(fmt_df)
    
    # Save
    upload_dir = os.path.join("uploads")
    os.makedirs(upload_dir, exist_ok=True)
    train_path = os.path.join(upload_dir, f"{session_id}_train.jsonl")
    holdout_path = os.path.join(upload_dir, f"{session_id}_holdout.jsonl")
    dataset_utils.save_jsonl(train_df, train_path)
    dataset_utils.save_jsonl(holdout_df, holdout_path)
    
    stats = DatasetStats(
        total_rows=len(fmt_df) + dups_removed + empty_removed,
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        duplicates_removed=dups_removed,
        empty_removed=empty_removed,
        avg_token_length=token_stats["avg_token_length"],
        max_token_length=token_stats["max_token_length"],
        min_token_length=token_stats["min_token_length"],
        p95_token_length=token_stats["p95_token_length"],
        column_names=list(df.columns),
        template_format=cfg["template_format"],
    )
    
    log_entry = f"[Dataset] Processed {stats['total_rows']} rows → {stats['train_rows']} train / {stats['holdout_rows']} holdout"
    
    return {
        "dataset": {
            "stats": stats,
            "train_path": train_path,
            "holdout_path": holdout_path,
            "sample_count": len(fmt_df),
            "avg_token_length": token_stats["avg_token_length"],
            "ready": True,
        },
        "agent_logs": [log_entry],
        "status": "dataset_ready",
    }
