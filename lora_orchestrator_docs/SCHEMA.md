# SCHEMA.md — LoRA Fine-Tuning Orchestrator
> All data models, TypedDicts, Pydantic schemas, and API contracts

---

## 1. LangGraph State — `OrchestrationState`

```python
# backend/app/state.py

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator


class RunConfig(TypedDict):
    """Immutable input config — set at pipeline start, never modified."""
    session_id:       str
    dataset_path:     str           # absolute path to uploaded file
    dataset_format:   str           # "csv" | "jsonl" | "huggingface"
    dataset_name:     str           # original filename or HF dataset name
    template_format:  str           # "alpaca" | "sharegpt"
    base_model:       str           # key from SUPPORTED_MODELS dict
    base_model_id:    str           # full HuggingFace model ID
    target_metric:    str           # "bleu" | "rouge_l" | "perplexity" | "exact_match"
    target_value:     float         # e.g. 0.60 — stop when primary metric >= this
    max_iterations:   int           # max Decision Agent retries (1–5)
    max_samples:      int           # cap on dataset rows
    quantization:     bool          # True = 4-bit bitsandbytes
    use_wandb:        bool
    use_mlflow:       bool


class DatasetStats(TypedDict):
    total_rows:          int
    train_rows:          int
    holdout_rows:        int
    duplicates_removed:  int
    empty_removed:       int
    avg_token_length:    float
    max_token_length:    int
    min_token_length:    int
    p95_token_length:    int
    column_names:        List[str]
    template_format:     str


class DatasetState(TypedDict):
    stats:           DatasetStats
    train_path:      str            # path to formatted JSONL for training
    holdout_path:    str            # path to formatted JSONL for evaluation
    sample_count:    int
    avg_token_length: float
    ready:           bool


class HyperparamConfig(TypedDict):
    """LoRA + training hyperparameters — one set per iteration."""
    # LoRA
    lora_r:           int           # rank: 4 | 8 | 16 | 32 | 64
    lora_alpha:       int           # alpha: typically 2x rank
    lora_dropout:     float         # 0.0 – 0.3
    target_modules:   List[str]     # e.g. ["q_proj", "v_proj", ...]
    bias:             str           # "none" | "all" | "lora_only"

    # Training
    learning_rate:    float         # 1e-5 – 5e-4
    num_epochs:       int           # 1 – 10
    batch_size:       int           # 1 | 2 | 4 | 8
    grad_accumulation: int          # gradient accumulation steps
    warmup_ratio:     float         # 0.0 – 0.1
    lr_scheduler:     str           # "cosine" | "linear" | "constant"
    optimizer:        str           # "paged_adamw_8bit" | "adamw_torch"
    weight_decay:     float         # 0.0 – 0.1
    max_grad_norm:    float         # 1.0
    max_seq_length:   int           # 512 | 1024 | 2048

    # Reasoning (from LLM)
    reasoning:        str           # why these values were chosen


class EpochLog(TypedDict):
    epoch:        int
    train_loss:   float
    val_loss:     float
    learning_rate: float
    elapsed_sec:  float


class TrainingState(TypedDict):
    logs:               List[EpochLog]   # per-epoch logs
    checkpoint_path:    str
    adapter_path:       str              # final saved adapter
    total_epochs_run:   int
    best_loss:          float
    best_epoch:         int
    stop_reason:        Optional[str]    # None | "divergence" | "plateau" | "completed"
    training_time_sec:  float


class EvalResults(TypedDict):
    bleu:           float
    rouge_l:        float
    perplexity:     float
    exact_match:    float            # 0.0 if not applicable
    primary_metric: str              # which metric was used as target
    primary_value:  float            # value of the primary metric
    eval_time_sec:  float
    samples_eval:   int


class DecisionRecord(TypedDict):
    iteration:       int
    decision:        str             # "stop" | "continue" | "adjust" | "fail"
    reasoning:       str
    metric_at_decision: float
    suggested_changes:  Optional[Dict[str, Any]]  # non-null if decision == "adjust"


class ReportState(TypedDict):
    json_path:      str
    md_path:        str
    adapter_path:   str             # final exported adapter directory


class OrchestrationState(TypedDict):
    """Master state — shared across all LangGraph nodes."""

    # ── Fixed input ────────────────────────────────────────────────
    run_config:           RunConfig

    # ── Agent outputs ──────────────────────────────────────────────
    dataset:              DatasetState
    current_hyperparams:  HyperparamConfig
    hyperparam_reasoning: str
    training:             TrainingState
    eval_results:         EvalResults

    # ── Accumulated history (operator.add → appended each iteration) ─
    eval_history:         Annotated[List[EvalResults], operator.add]
    hyperparam_history:   Annotated[List[HyperparamConfig], operator.add]
    decision_history:     Annotated[List[DecisionRecord], operator.add]
    training_logs:        Annotated[List[EpochLog], operator.add]
    agent_logs:           Annotated[List[str], operator.add]

    # ── Decision control ───────────────────────────────────────────
    decision:             str        # latest: "stop"|"continue"|"adjust"|"fail"
    decision_reasoning:   str
    suggested_hyperparam_changes: Optional[Dict[str, Any]]
    iteration:            int        # starts at 0, increments on each loop

    # ── Report ─────────────────────────────────────────────────────
    report:               ReportState

    # ── Pipeline control ───────────────────────────────────────────
    status:               str        # "initializing"|"running"|"done"|"failed"
    current_agent:        str
    error:                Optional[str]
```

---

## 2. FastAPI Request / Response Models

```python
# backend/app/models.py (Pydantic)

from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    session_id:  str
    file_path:   str
    file_name:   str
    file_size_mb: float


class StartRequest(BaseModel):
    session_id:      str
    file_path:       str
    file_name:       str
    template_format: str = "alpaca"
    base_model:      str = "qwen2.5-7b"
    target_metric:   str = "rouge_l"
    target_value:    float = Field(0.60, ge=0.0, le=1.0)
    max_iterations:  int   = Field(3, ge=1, le=5)
    max_samples:     int   = Field(10000, ge=100, le=500000)
    quantization:    bool  = True
    use_wandb:       bool  = False
    use_mlflow:      bool  = False
    llm_provider:    str   = "openai"
    llm_model:       Optional[str] = None


class StartResponse(BaseModel):
    session_id:   str
    stream_url:   str            # /api/stream/{session_id}
    status:       str


class HealthResponse(BaseModel):
    status:       str
    gpu_available: bool
    vram_gb:      Optional[float]


class ModelsResponse(BaseModel):
    models: list[dict]           # [{key, name, params, vram_required_gb}]
```

---

## 3. Dataset Formats

### Alpaca Template

```python
# Input row (CSV/JSONL must have these columns):
# instruction, input (optional), output

# Formatted output:
{
    "text": (
        "Below is an instruction that describes a task"
        "{input_section}. Write a response.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )
}

# Where input_section:
# - if input field is non-empty: ", paired with an input that provides further context"
# - if input field is empty: "" (omit entirely)
```

### ShareGPT Template

```python
# Input row must have "conversations" field:
# [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]

# Formatted output — TRL's DataCollatorForCompletionOnlyLM handles this:
{
    "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt",   "value": "..."}
    ]
}
```

---

## 4. SSE Event Schemas

```python
# All SSE events are JSON strings. Every event has `type` as the first key.

AgentStartEvent = {
    "type":    "agent_start",
    "agent":   str,       # "dataset_agent" | "hyperparam_agent" | etc.
    "message": str,       # human-readable status
    "ts":      str,       # ISO timestamp
}

AgentCompleteEvent = {
    "type":  "agent_complete",
    "agent": str,
    "data":  dict,        # agent-specific summary (DatasetStats, HyperparamConfig, etc.)
    "ts":    str,
}

TrainingLogEvent = {
    "type":       "training_log",
    "epoch":      int,
    "train_loss": float,
    "val_loss":   float,
    "lr":         float,
    "elapsed_sec": float,
    "ts":         str,
}

EarlyStopEvent = {
    "type":   "early_stop",
    "reason": str,        # "divergence" | "plateau"
    "epoch":  int,
    "ts":     str,
}

EvalCompleteEvent = {
    "type":        "eval_complete",
    "bleu":        float,
    "rouge_l":     float,
    "perplexity":  float,
    "exact_match": float,
    "primary_metric": str,
    "primary_value":  float,
    "ts":          str,
}

DecisionEvent = {
    "type":       "decision",
    "decision":   str,    # "stop" | "continue" | "adjust" | "fail"
    "reasoning":  str,
    "iteration":  int,
    "ts":         str,
}

CompleteEvent = {
    "type":        "complete",
    "adapter_path": str,
    "report_url":   str,
    "final_metric": float,
    "total_iterations": int,
    "ts":          str,
}

ErrorEvent = {
    "type":    "error",
    "message": str,
    "agent":   str,
    "ts":      str,
}
```

---

## 5. Experiment Report Schema (JSON)

```python
# outputs/reports/{session_id}/report.json

ExperimentReport = {
    "session_id":     str,
    "created_at":     str,           # ISO timestamp
    "status":         str,           # "success" | "failed"

    "run_config":     RunConfig,

    "dataset_summary": {
        "name":            str,
        "total_rows":      int,
        "train_rows":      int,
        "holdout_rows":    int,
        "avg_token_length": float,
        "template_used":   str,
    },

    "iterations": [
        {
            "iteration":     int,
            "hyperparams":   HyperparamConfig,
            "training": {
                "epochs_run":   int,
                "best_loss":    float,
                "stop_reason":  str | None,
                "training_time_sec": float,
                "loss_curve":   List[EpochLog],
            },
            "eval_results":  EvalResults,
            "decision":      DecisionRecord,
        }
    ],

    "final_results": {
        "best_iteration":     int,
        "best_hyperparams":   HyperparamConfig,
        "final_eval":         EvalResults,
        "adapter_path":       str,
        "total_training_time_sec": float,
    },

    "agent_logs": List[str],
}
```

---

## 6. Hyperparameter Constraints

```python
# Validation rules — enforce in hyperparam_agent.py

VALID_RANKS        = {4, 8, 16, 32, 64}
LR_RANGE           = (1e-5, 5e-4)
DROPOUT_RANGE      = (0.0, 0.3)
EPOCH_RANGE        = (1, 10)
BATCH_SIZES        = {1, 2, 4, 8}
GRAD_ACCUM_RANGE   = (1, 16)
WARMUP_RANGE       = (0.0, 0.1)
MAX_SEQ_LENGTHS    = {512, 1024, 2048}
VALID_SCHEDULERS   = {"cosine", "linear", "constant"}
VALID_OPTIMIZERS   = {"paged_adamw_8bit", "adamw_torch"}
VALID_BIASES       = {"none", "all", "lora_only"}

# alpha rule: typically 2x rank, always integer
# lora_alpha = lora_r * 2 is the safe default
```

---

## 7. Early Stopping Conditions

```python
# Checked in training_agent.py after every epoch

def check_early_stop(logs: List[EpochLog]) -> Optional[str]:
    if len(logs) < 2:
        return None

    # Divergence: loss > 5.0 for 2 consecutive epochs
    if len(logs) >= 2:
        last_two = logs[-2:]
        if all(e["train_loss"] > 5.0 for e in last_two):
            return "divergence"

    # Plateau: loss delta < 0.001 for 3 consecutive epochs
    if len(logs) >= 3:
        last_three = logs[-3:]
        deltas = [
            abs(last_three[i]["train_loss"] - last_three[i-1]["train_loss"])
            for i in range(1, 3)
        ]
        if all(d < 0.001 for d in deltas):
            return "plateau"

    return None
```
