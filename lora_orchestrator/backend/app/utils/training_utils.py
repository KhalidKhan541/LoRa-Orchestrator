import time
from typing import Optional, List
from transformers import TrainingArguments, TrainerCallback


def build_training_args(hyperparams: dict, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hyperparams["num_epochs"],
        per_device_train_batch_size=hyperparams["batch_size"],
        gradient_accumulation_steps=hyperparams["grad_accumulation"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        max_grad_norm=hyperparams["max_grad_norm"],
        warmup_ratio=hyperparams["warmup_ratio"],
        lr_scheduler_type=hyperparams["lr_scheduler"],
        optim=hyperparams["optimizer"],
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


class EpochLogCallback(TrainerCallback):
    def __init__(self, sse_callback=None):
        self.epoch_logs: list = []
        self.sse_callback = sse_callback
        self._start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        elapsed = time.time() - self._start_time if self._start_time else 0
        train_loss = logs.get("train_loss", 0.0) if logs else 0.0
        val_loss = logs.get("eval_loss", train_loss) if logs else train_loss
        lr = logs.get("learning_rate", 0.0) if logs else 0.0
        epoch_log = {
            "epoch": int(state.epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr,
            "elapsed_sec": elapsed,
        }
        self.epoch_logs.append(epoch_log)
        if self.sse_callback:
            self.sse_callback(epoch_log)


def check_early_stop(logs: List[dict]) -> Optional[str]:
    if len(logs) < 2:
        return None
    last_two = logs[-2:]
    if all(e["train_loss"] > 5.0 for e in last_two):
        return "divergence"
    if len(logs) >= 3:
        last_three = logs[-3:]
        deltas = [abs(last_three[i]["train_loss"] - last_three[i - 1]["train_loss"]) for i in range(1, 3)]
        if all(d < 0.001 for d in deltas):
            return "plateau"
    return None
