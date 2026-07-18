import json
from langchain_core.messages import SystemMessage, HumanMessage
from backend.app.utils import model_utils

SUPPORTED_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
}

SYSTEM_PROMPT = """You are a senior ML engineer specializing in LoRA fine-tuning of large language models.
Your job is to select optimal hyperparameters for a LoRA fine-tuning run.

You reason carefully about the dataset size, base model architecture, available VRAM,
and previous iteration results before selecting values.

HYPERPARAMETER RULES (enforce strictly — the training pipeline validates these):
- lora_r MUST be one of: 4, 8, 16, 32, 64
- lora_alpha MUST be an integer — recommended value: lora_r * 2
- lora_dropout MUST be in range [0.0, 0.3]
- learning_rate MUST be in range [1e-5, 5e-4]
- num_epochs MUST be in range [1, 10]
- batch_size MUST be one of: 1, 2, 4, 8
- lr_scheduler MUST be one of: "cosine", "linear", "constant"
- optimizer MUST be one of: "paged_adamw_8bit", "adamw_torch"
- bias MUST be one of: "none", "all", "lora_only"
- max_seq_length MUST be one of: 512, 1024, 2048

SELECTION GUIDELINES:
- For small datasets (< 1000 rows): use lower rank (8–16), more epochs (5–8), higher dropout (0.1–0.2) to prevent overfitting
- For large datasets (> 10000 rows): higher rank (16–32) is appropriate, fewer epochs (2–4), lower dropout (0.0–0.05)
- For 7B models with 4-bit quantization: batch_size=2, grad_accumulation=4 is a safe starting point for 16GB VRAM
- cosine scheduler: best general choice for instruction tuning
- paged_adamw_8bit: required when quantization=True, saves VRAM
- If previous iteration diverged (loss > 5): lower lr by 5-10x, reduce rank
- If previous iteration plateaued early: increase lr by 2x, add warmup
- If eval metric is close to target (within 0.1): small incremental changes only
- If eval metric is far from target (> 0.2 gap): make bolder changes

Respond with ONLY a valid JSON object. No explanation before or after the JSON.
No markdown fences. The JSON must exactly match this schema:

{
  "lora_r": int,
  "lora_alpha": int,
  "lora_dropout": float,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "bias": "none",
  "learning_rate": float,
  "num_epochs": int,
  "batch_size": int,
  "grad_accumulation": int,
  "warmup_ratio": float,
  "lr_scheduler": "cosine",
  "optimizer": "paged_adamw_8bit",
  "weight_decay": float,
  "max_grad_norm": 1.0,
  "max_seq_length": int,
  "reasoning": "2-3 sentence explanation of why these values were chosen"
}"""


def build_hyperparam_message(state: dict) -> str:
    cfg = state["run_config"]
    ds = state.get("dataset", {})
    stats = ds.get("stats", {})
    itr = state.get("iteration", 0)
    hist = state.get("eval_history", [])
    hp_h = state.get("hyperparam_history", [])
    changes = state.get("suggested_hyperparam_changes", {})

    base = f"""
DATASET INFORMATION:
- Name: {cfg['dataset_name']}
- Training rows: {stats.get('train_rows', 0)}
- Avg token length: {stats.get('avg_token_length', 0):.0f}
- P95 token length: {stats.get('p95_token_length', 0)}
- Template: {cfg['template_format']}

BASE MODEL:
- Model: {cfg['base_model_id']}
- Quantization: {'4-bit (bitsandbytes)' if cfg['quantization'] else 'None (full precision)'}

TARGET:
- Metric: {cfg['target_metric']}
- Target value: {cfg['target_value']}

ITERATION: {itr + 1} of {cfg['max_iterations']}
"""
    if itr == 0:
        base += "\nThis is the FIRST iteration. Select a sensible baseline configuration."
    else:
        if hist and hp_h:
            prev_eval = hist[-1]
            prev_hp = hp_h[-1]
            base += f"""
PREVIOUS ITERATION RESULTS:
- Hyperparameters used: {json.dumps(prev_hp, indent=2)}
- Eval results: {json.dumps(prev_eval, indent=2)}
- Primary metric ({cfg['target_metric']}): {prev_eval.get('primary_value', 0):.4f}
- Target: {cfg['target_value']} — {"MET" if prev_eval.get('primary_value', 0) >= cfg['target_value'] else "NOT MET"}
- Training stop reason: {state.get('training', {}).get('stop_reason', 'completed normally')}
"""
            if changes:
                base += f"\nSuggested changes from Decision Agent: {json.dumps(changes, indent=2)}"
            base += "\nSelect NEW hyperparameters that address the issues above."
    return base


DEFAULT_HYPERPARAMS = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 2,
    "grad_accumulation": 4,
    "warmup_ratio": 0.05,
    "lr_scheduler": "cosine",
    "optimizer": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_seq_length": 1024,
    "reasoning": "Default baseline configuration for 7B model with 4-bit quantization.",
}


def _validate_hyperparams(hp: dict) -> dict:
    validated = dict(DEFAULT_HYPERPARAMS)
    validated.update(hp)
    if validated["lora_r"] not in {4, 8, 16, 32, 64}:
        validated["lora_r"] = 16
    validated["lora_alpha"] = validated["lora_r"] * 2
    validated["lora_dropout"] = max(0.0, min(0.3, validated["lora_dropout"]))
    validated["learning_rate"] = max(1e-5, min(5e-4, validated["learning_rate"]))
    validated["num_epochs"] = max(1, min(10, validated["num_epochs"]))
    if validated["batch_size"] not in {1, 2, 4, 8}:
        validated["batch_size"] = 2
    validated["grad_accumulation"] = max(1, min(16, validated["grad_accumulation"]))
    validated["warmup_ratio"] = max(0.0, min(0.1, validated["warmup_ratio"]))
    if validated["lr_scheduler"] not in {"cosine", "linear", "constant"}:
        validated["lr_scheduler"] = "cosine"
    if validated["optimizer"] not in {"paged_adamw_8bit", "adamw_torch"}:
        validated["optimizer"] = "paged_adamw_8bit"
    if validated["bias"] not in {"none", "all", "lora_only"}:
        validated["bias"] = "none"
    if validated["max_seq_length"] not in {512, 1024, 2048}:
        validated["max_seq_length"] = 1024
    return validated


def hyperparam_node(state: dict, llm=None) -> dict:
    cfg = state["run_config"]
    base_model = cfg.get("base_model", "qwen2.5-7b")
    model_family = model_utils.get_model_family(base_model)

    if llm is None:
        hp = dict(DEFAULT_HYPERPARAMS)
        hp["target_modules"] = model_utils.TARGET_MODULES.get(model_family, DEFAULT_HYPERPARAMS["target_modules"])
        reasoning = "LLM not available; using default hyperparameters."
    else:
        user_msg = build_hyperparam_message(state)
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            hp = json.loads(raw)
        except Exception:
            hp = dict(DEFAULT_HYPERPARAMS)
            reasoning = "Failed to parse LLM response; using defaults."
            hp["reasoning"] = reasoning

        # Apply suggested changes from decision agent
        changes = state.get("suggested_hyperparam_changes")
        if changes:
            hp.update(changes)

        hp = _validate_hyperparams(hp)
        hp["target_modules"] = model_utils.TARGET_MODULES.get(model_family, hp.get("target_modules", []))
        reasoning = hp.get("reasoning", "Selected by LLM.")

    log_entry = f"[Hyperparam] Selected rank={hp['lora_r']}, lr={hp['learning_rate']} (reasoning: {hp.get('reasoning', '')[:80]}...)"

    return {
        "current_hyperparams": hp,
        "hyperparam_reasoning": hp.get("reasoning", ""),
        "agent_logs": [log_entry],
    }
