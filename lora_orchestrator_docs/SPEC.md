# SPEC.md — LoRA Fine-Tuning Orchestrator
> Architecture, file structure, and technical contracts

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Vanilla JS)                     │
│  Upload → Configure → [Start] → Live Dashboard → Report         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP + SSE
┌───────────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend (main.py)                      │
│  POST /api/start  ·  GET /api/stream/{id}  ·  GET /api/report   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ graph.invoke() / graph.stream()
┌───────────────────────────▼─────────────────────────────────────┐
│                  LangGraph StateGraph Pipeline                    │
│                                                                   │
│  ┌───────────┐   ┌──────────────┐   ┌──────────┐               │
│  │  Dataset  │──▶│Hyperparam    │──▶│ Training │               │
│  │  Agent    │   │Agent         │   │  Agent   │               │
│  └───────────┘   └──────────────┘   └────┬─────┘               │
│                                          │                       │
│                                    ┌─────▼──────┐               │
│                                    │ Evaluation │               │
│                                    │   Agent    │               │
│                                    └─────┬──────┘               │
│                                          │                       │
│                                    ┌─────▼──────┐               │
│                                    │  Decision  │◀──┐           │
│                                    │   Agent    │   │           │
│                                    └─────┬──────┘   │           │
│                                          │           │           │
│                              ┌───────────┼───────────┘           │
│                              │           │                       │
│                           STOP/FAIL   ADJUST/CONTINUE            │
│                              │           │                       │
│                           REPORT    Hyperparam Agent             │
└──────────────────────────────┼───────────┴─────────────────────-┘
                               │
                    ┌──────────▼──────────┐
                    │   outputs/           │
                    │   ├── adapter/       │
                    │   ├── reports/       │
                    │   └── checkpoints/   │
                    └─────────────────────┘
```

---

## 2. Directory Structure

Implement **exactly** this structure. No deviations.

```
lora_orchestrator/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── state.py                  # OrchestrationState TypedDict
│   │   ├── graph.py                  # StateGraph: nodes + edges + routing
│   │   ├── llm.py                    # Orchestration LLM factory
│   │   │
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_agent.py      # Load → clean → format → validate
│   │   │   ├── hyperparam_agent.py   # LLM selects LoRA hyperparameters
│   │   │   ├── training_agent.py     # PEFT + TRL SFTTrainer execution
│   │   │   ├── eval_agent.py         # BLEU, ROUGE, perplexity, task eval
│   │   │   └── decision_agent.py     # STOP / CONTINUE / ADJUST / FAIL
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── dataset_utils.py      # Alpaca/ShareGPT formatters, validators
│   │       ├── training_utils.py     # SFTTrainer config builder, callbacks
│   │       ├── eval_utils.py         # BLEU/ROUGE/perplexity scorers
│   │       ├── model_utils.py        # Model loader, quantization, adapter export
│   │       └── report_utils.py       # Markdown + JSON report generator
│   │
│   ├── main.py                       # FastAPI app + SSE streaming
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js                    # Main controller
│       ├── upload.js                 # File upload + dataset preview
│       ├── config.js                 # Model/hyperparameter config form
│       ├── stream.js                 # SSE listener + agent state machine
│       ├── charts.js                 # Real-time loss chart (Chart.js)
│       └── report.js                 # Report panel renderer
│
├── outputs/
│   ├── adapters/                     # Saved LoRA adapters per run
│   ├── reports/                      # Experiment reports per run
│   └── checkpoints/                  # Intermediate training checkpoints
│
├── uploads/                          # Raw uploaded datasets
├── .env.example
└── README.md
```

---

## 3. LangGraph Graph Definition

```python
# backend/app/graph.py

g = StateGraph(OrchestrationState)

# Nodes
g.add_node("dataset_agent",    lambda s: dataset_node(s, llm))
g.add_node("hyperparam_agent", lambda s: hyperparam_node(s, llm))
g.add_node("training_agent",   training_node)           # no LLM needed
g.add_node("eval_agent",       eval_node)               # no LLM needed
g.add_node("decision_agent",   lambda s: decision_node(s, llm))
g.add_node("report_agent",     report_node)             # no LLM needed

# Fixed edges
g.set_entry_point("dataset_agent")
g.add_edge("dataset_agent",    "hyperparam_agent")
g.add_edge("hyperparam_agent", "training_agent")
g.add_edge("training_agent",   "eval_agent")
g.add_edge("eval_agent",       "decision_agent")

# Conditional routing from Decision Agent
g.add_conditional_edges(
    "decision_agent",
    route_decision,
    {
        "adjust":   "hyperparam_agent",   # loop: new hyperparams → retrain
        "continue": "training_agent",     # loop: more epochs same config
        "stop":     "report_agent",       # success
        "fail":     "report_agent",       # max retries exceeded
    }
)

g.add_edge("report_agent", END)
```

---

## 4. Node Specifications

### 4.1 Dataset Agent (`nodes/dataset_agent.py`)

**Inputs from state:** `run_config.dataset_path`, `run_config.dataset_format`,
`run_config.template_format`, `run_config.max_samples`

**Logic:**
1. Load dataset: CSV → pandas, JSONL → json.loads per line,
   HuggingFace name → `datasets.load_dataset()`
2. Deduplicate on the instruction field
3. Filter rows where instruction or output is empty / < 10 chars
4. Format each row into the chosen template (Alpaca or ShareGPT)
5. Compute dataset stats: total rows, avg token length, length distribution,
   label balance (for classification tasks)
6. Split: 90% train / 10% holdout (stratified if classification)
7. Save formatted dataset to `uploads/{session_id}_train.jsonl` and `_holdout.jsonl`
8. Return dataset stats and file paths

**State keys written:**
- `dataset.stats` — DatasetStats
- `dataset.train_path` — str
- `dataset.holdout_path` — str
- `dataset.sample_count` — int
- `dataset.avg_token_length` — float
- `agent_logs` — append log entry
- `status` → `"dataset_ready"`

---

### 4.2 Hyperparameter Agent (`nodes/hyperparam_agent.py`)

**Inputs from state:** `dataset.stats`, `run_config.base_model`,
`run_config.target_metric`, `run_config.target_value`,
`iteration_history` (previous hyperparams + results)

**Logic:**
1. Build context: dataset size, avg token length, base model name,
   previous iteration results if iteration > 0
2. Call orchestration LLM with HYPERPARAM_SELECTION_PROMPT
3. Parse JSON response into HyperparamConfig
4. Validate: rank must be power of 2 (4/8/16/32/64),
   lr must be in [1e-5, 5e-4], epochs in [1, 10]
5. Log reasoning string

**State keys written:**
- `current_hyperparams` — HyperparamConfig
- `hyperparam_reasoning` — str
- `agent_logs` — append

---

### 4.3 Training Agent (`nodes/training_agent.py`)

**Inputs from state:** `current_hyperparams`, `dataset.train_path`,
`run_config.base_model`, `run_config.quantization`, `session_id`

**Logic:**
1. Load base model via `model_utils.load_model()` with 4-bit quantization
2. Apply PEFT LoRA config from `current_hyperparams`
3. Build `SFTTrainer` with custom `EpochLogCallback` that:
   - After each epoch: appends `{epoch, train_loss, val_loss, lr}` to state
   - Publishes SSE event `training_log` with the epoch dict
4. Run `trainer.train()`
5. Detect early stopping conditions:
   - Loss divergence: `train_loss > 5.0` for 2 consecutive epochs
   - Plateau: loss delta < 0.001 for 3 consecutive epochs
   - If either: set `training.stop_reason` and stop
6. Save checkpoint to `outputs/checkpoints/{session_id}/`
7. Return training logs and checkpoint path

**State keys written:**
- `training.logs` — List[EpochLog] (accumulated)
- `training.checkpoint_path` — str
- `training.total_epochs_run` — int
- `training.stop_reason` — str | None ("divergence" | "plateau" | "completed")
- `training.best_loss` — float
- `agent_logs` — append

---

### 4.4 Evaluation Agent (`nodes/eval_agent.py`)

**Inputs from state:** `training.checkpoint_path`, `dataset.holdout_path`,
`run_config.target_metric`, `current_hyperparams`

**Logic:**
1. Load the fine-tuned adapter from checkpoint
2. Run inference on holdout set (batch_size=4)
3. Compute metrics:
   - `bleu` — corpus BLEU on generated vs reference outputs
   - `rouge_l` — ROUGE-L F1
   - `perplexity` — mean per-token perplexity on holdout
   - `exact_match` — % of outputs matching reference exactly (for classification)
4. Store all metrics in eval_results
5. Pick the `primary_metric` value based on `run_config.target_metric`

**State keys written:**
- `eval_results` — EvalResults
- `eval_history` — append (accumulated across iterations)
- `agent_logs` — append

---

### 4.5 Decision Agent (`nodes/decision_agent.py`)

**Inputs from state:** `eval_results`, `eval_history`, `current_hyperparams`,
`iteration`, `run_config.target_metric`, `run_config.target_value`,
`run_config.max_iterations`

**Logic:**
1. Check termination conditions:
   - `iteration >= max_iterations` → decision: "fail"
   - `primary_metric >= target_value` → decision: "stop"
   - `training.stop_reason == "divergence"` → decision: "fail" (don't retry diverged)
2. If neither: call orchestration LLM with DECISION_PROMPT
3. Parse LLM decision: "adjust" or "continue"
4. If "adjust": LLM also returns `suggested_changes` dict
5. Log full decision reasoning

**State keys written:**
- `decision` — "stop" | "continue" | "adjust" | "fail"
- `decision_reasoning` — str
- `suggested_hyperparam_changes` — Dict | None
- `iteration` — increment by 1
- `agent_logs` — append

---

### 4.6 Report Agent (`nodes/report_agent.py`)

**Inputs from state:** All state fields

**Logic:**
1. Export final LoRA adapter to `outputs/adapters/{session_id}/`
2. Generate JSON experiment report (full state serialization)
3. Generate Markdown report via `report_utils.build_report()`
4. Save both to `outputs/reports/{session_id}/`
5. Set `status = "done"` or `status = "failed"`

**State keys written:**
- `report.json_path` — str
- `report.md_path` — str
- `report.adapter_path` — str
- `status` — "done" | "failed"

---

## 5. FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload dataset file → `{session_id, file_path}` |
| POST | `/api/start` | Start pipeline → begins SSE stream |
| GET | `/api/stream/{session_id}` | SSE event stream for live updates |
| GET | `/api/report/{session_id}` | Get final Markdown report |
| GET | `/api/report/{session_id}/json` | Get full JSON experiment report |
| GET | `/api/adapter/{session_id}` | Download adapter as .zip |
| GET | `/api/health` | Health check |
| GET | `/api/models` | List supported base models |

### SSE Event Types

```jsonc
// Agent lifecycle
{ "type": "agent_start",    "agent": "dataset_agent",  "message": "Loading dataset..." }
{ "type": "agent_complete", "agent": "dataset_agent",  "data": { ...DatasetStats } }

// Training loop (fires every epoch)
{ "type": "training_log",   "epoch": 3, "train_loss": 1.23, "val_loss": 1.45, "lr": 0.0002 }

// Early stopping
{ "type": "early_stop",     "reason": "plateau", "epoch": 7 }

// Decision
{ "type": "decision",       "decision": "adjust", "reasoning": "...", "iteration": 2 }

// Evaluation
{ "type": "eval_complete",  "bleu": 0.42, "rouge_l": 0.61, "perplexity": 12.3 }

// Done
{ "type": "complete",       "adapter_path": "...", "report_url": "/api/report/..." }
{ "type": "error",          "message": "...", "agent": "training_agent" }
```

---

## 6. Routing Function

```python
def route_decision(state: OrchestrationState) -> str:
    decision = state["decision"]
    iteration = state["iteration"]
    max_iter  = state["run_config"]["max_iterations"]

    if decision == "stop":
        return "stop"
    if decision == "fail" or iteration >= max_iter:
        return "fail"
    if decision == "adjust":
        return "adjust"
    if decision == "continue":
        return "continue"
    return "fail"  # safety fallback
```

---

## 7. Supported Base Models

```python
SUPPORTED_MODELS = {
    "qwen2.5-7b":      "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b":     "Qwen/Qwen2.5-14B-Instruct",
    "llama-3-8b":      "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b":      "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini":      "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2-9b":      "google/gemma-2-9b-it",
}

TARGET_MODULES = {
    "qwen2.5":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama":     ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi":       ["q_proj", "k_proj", "v_proj", "dense"],
    "gemma":     ["q_proj", "k_proj", "v_proj", "o_proj"],
}
```

---

## 8. Requirements

```
# LangGraph + LangChain
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.3.0
langchain-groq>=0.2.0
langchain-ollama>=0.2.0

# Training
torch>=2.1.0
transformers>=4.45.0
peft>=0.13.0
trl>=0.12.0
bitsandbytes>=0.44.0
accelerate>=0.34.0
datasets>=2.20.0
sentencepiece>=0.2.0
tokenizers>=0.20.0

# Evaluation
nltk>=3.9.0
rouge-score>=0.1.2
evaluate>=0.4.3

# API
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.0.0
python-multipart>=0.0.12

# Utilities
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
aiofiles>=23.0.0
tqdm>=4.66.0
psutil>=5.9.0

# Optional tracking
wandb>=0.18.0
mlflow>=2.16.0
```
