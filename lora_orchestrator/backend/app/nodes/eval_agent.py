import os
import gc
import time
import torch
from datasets import load_dataset as hf_load_dataset
from backend.app.utils import model_utils, eval_utils


def eval_node(state: dict) -> dict:
    cfg = state["run_config"]
    hp = state["current_hyperparams"]
    training = state.get("training", {})

    checkpoint_path = training.get("checkpoint_path", "")
    adapter_path = training.get("adapter_path", "")
    holdout_path = state["dataset"].get("holdout_path", "")

    model, tokenizer = model_utils.load_adapter(cfg["base_model_id"], adapter_path, cfg["quantization"])

    dataset = hf_load_dataset("json", data_files=holdout_path, split="train")

    prompts = []
    references = []
    for item in dataset:
        if cfg["template_format"] == "alpaca":
            instruction = item.get("text", "")
            # Extract instruction and response from formatted text
            parts = instruction.split("### Response:\n")
            if len(parts) == 2:
                prompts.append(parts[0] + "### Response:\n")
                references.append(parts[1])
            else:
                prompts.append(instruction)
                references.append("")
        else:
            convs = item.get("conversations", [])
            if convs:
                prompts.append(convs[0].get("value", ""))
                references.append(convs[-1].get("value", "") if len(convs) > 1 else "")

    start_time = time.time()
    predictions = eval_utils.run_inference(model, tokenizer, prompts, batch_size=4)
    eval_time = time.time() - start_time

    bleu = eval_utils.compute_bleu(predictions, references)
    rouge_l = eval_utils.compute_rouge_l(predictions, references)
    perplexity = eval_utils.compute_perplexity(model, tokenizer, references)
    exact_match = eval_utils.compute_exact_match(predictions, references)

    target_metric = cfg.get("target_metric", "rouge_l")
    metric_map = {"bleu": bleu, "rouge_l": rouge_l, "perplexity": perplexity, "exact_match": exact_match}
    primary_value = metric_map.get(target_metric, rouge_l)

    eval_results = {
        "bleu": bleu,
        "rouge_l": rouge_l,
        "perplexity": perplexity,
        "exact_match": exact_match,
        "primary_metric": target_metric,
        "primary_value": primary_value,
        "eval_time_sec": eval_time,
        "samples_eval": len(prompts),
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "eval_results": eval_results,
        "eval_history": [eval_results],
        "agent_logs": [f"[Eval] BLEU={bleu:.4f}, ROUGE-L={rouge_l:.4f}, PPL={perplexity:.2f}, EM={exact_match:.4f}"],
    }
