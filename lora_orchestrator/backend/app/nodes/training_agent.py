import os
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset as hf_load_dataset
from backend.app.utils import model_utils, training_utils


def training_node(state: dict) -> dict:
    cfg = state["run_config"]
    hp = state["current_hyperparams"]
    session_id = cfg["session_id"]
    iteration = state.get("iteration", 0)

    model, tokenizer = model_utils.load_model(cfg["base_model_id"], cfg["quantization"])
    model = model_utils.apply_lora(model, hp, model_utils.get_model_family(cfg["base_model"]))

    dataset = hf_load_dataset("json", data_files=state["dataset"]["train_path"], split="train")

    output_dir = os.path.join("outputs", "checkpoints", session_id, f"iter_{iteration}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = training_utils.build_training_args(hp, output_dir)

    epoch_logs = []

    def sse_callback(log):
        epoch_logs.append(log)

    epoch_log_cb = training_utils.EpochLogCallback(sse_callback=sse_callback)

    def tokenize_fn(examples):
        if "text" in examples:
            return tokenizer(examples["text"], truncation=True, max_length=hp["max_seq_length"], padding="max_length")
        return tokenizer(examples.get("conversations", [""]), truncation=True, max_length=hp["max_seq_length"], padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        callbacks=[epoch_log_cb],
    )

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # Check early stopping
    stop_reason = training_utils.check_early_stop(epoch_logs)
    if stop_reason is None:
        stop_reason = "completed"

    best_loss = min((log["train_loss"] for log in epoch_logs), default=0.0)
    best_epoch = next(
        (log["epoch"] for log in epoch_logs if log["train_loss"] == best_loss),
        0,
    )

    model_utils.save_adapter(model, os.path.join(output_dir, "adapter"))

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "training": {
            "logs": epoch_logs,
            "checkpoint_path": output_dir,
            "adapter_path": os.path.join(output_dir, "adapter"),
            "total_epochs_run": len(epoch_logs),
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "stop_reason": stop_reason,
            "training_time_sec": total_time,
        },
        "training_logs": epoch_logs,
        "agent_logs": [f"[Training] Completed {len(epoch_logs)} epochs in {total_time:.1f}s, best_loss={best_loss:.4f}"],
    }
