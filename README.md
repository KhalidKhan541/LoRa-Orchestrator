# ⚡ LoRa Fine-Tuning Orchestrator

An autonomous, LangGraph-powered pipeline that orchestrates the entire LoRA fine-tuning lifecycle — from raw dataset ingestion to trained adapter export — without requiring human intervention. The system acts like a senior ML engineer: it curates the dataset, selects hyperparameters, launches training, evaluates quality, decides whether to stop early or adjust, and produces a final trained LoRA adapter alongside a structured experiment report.

---

## What It Does

```
Upload Dataset → Agent Selects Hyperparams → Training Runs → Eval Scores Computed
     ↓                    ↓                        ↓                  ↓
  Clean & Format     LLM picks rank,        Loss streamed         BLEU / ROUGE-L
  into Alpaca/        lr, epochs, etc.       live to dashboard     / PPL / Exact Match
  ShareGPT                                                            
                                                    ↓
                                              Decision Agent:
                                              STOP / CONTINUE / ADJUST
                                                    ↓
                                           Report + Adapter Export
```

### 5 Specialist Agents

| Agent | Role |
|-------|------|
| **Dataset Agent** | Loads raw data (CSV/JSONL/HuggingFace), deduplicates, filters empty rows, formats into Alpaca or ShareGPT templates, computes token statistics, splits 90/10 train/holdout |
| **Hyperparameter Agent** | LLM-driven selection of LoRA rank, alpha, dropout, learning rate, scheduler, optimizer — adapts based on dataset size, model architecture, and previous iteration results |
| **Training Agent** | Executes LoRA fine-tuning via HuggingFace PEFT + TRL SFTTrainer with 4-bit quantization, custom epoch logging, and automatic early stopping on divergence or plateau |
| **Evaluation Agent** | Runs inference on holdout set and computes BLEU, ROUGE-L, perplexity, and exact match against reference outputs |
| **Decision Agent** | Analyzes metric trends and decides: STOP (target met), CONTINUE (more epochs), ADJUST (new hyperparams), or FAIL (escalate to human) |

---

## Quickstart

### Prerequisites

- Python 3.11+
- CUDA GPU with ≥16GB VRAM (24GB recommended)
- An LLM API key (OpenAI, Anthropic, Groq, or Ollama running locally)

### Installation

```bash
# Clone the repo
git clone https://github.com/KhalidKhan541/LoRa-Orchestrator.git
cd LoRa-Orchestrator

# Install dependencies
pip install -r lora_orchestrator/backend/requirements.txt

# Configure environment
cp lora_orchestrator/.env.example lora_orchestrator/.env
# Edit .env — add your API key and set LLM_PROVIDER
```

### Run

```bash
# Start the backend
cd lora_orchestrator
uvicorn backend.main:app --reload --port 8000

# Open the frontend
# Navigate to frontend/index.html in your browser
```

1. **Upload** a dataset (.csv, .jsonl, or .json)
2. **Configure** base model, template, target metric, and max iterations
3. **Click Start** — watch the pipeline run live with loss curves updating in real-time
4. **Download** the trained LoRA adapter and experiment report when done

---

## Supported Base Models

| Key | Model | Params | VRAM Required |
|-----|-------|--------|---------------|
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct | 7B | ~16 GB |
| `qwen2.5-14b` | Qwen/Qwen2.5-14B-Instruct | 14B | ~24 GB |
| `llama-3-8b` | meta-llama/Meta-Llama-3-8B-Instruct | 8B | ~16 GB |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | 7B | ~16 GB |
| `phi-3-mini` | microsoft/Phi-3-mini-4k-instruct | 3.8B | ~8 GB |
| `gemma-2-9b` | google/gemma-2-9b-it | 9B | ~16 GB |

All models are loaded with **4-bit quantization** (bitsandbytes NF4) by default to fit within VRAM limits.

---

## Dataset Formats

### Alpaca Format (Recommended)

```json
{
  "instruction": "Summarize the following article about climate change.",
  "input": "Global temperatures have risen by 1.1°C since pre-industrial times...",
  "output": "The article discusses the 1.1°C rise in global temperatures since pre-industrial times and its implications."
}
```

- Fields: `instruction` (required), `input` (optional), `output` (required)
- Formatted into: `### Instruction:\n{instruction}\n\n### Response:\n{output}`

### ShareGPT Format

```json
{
  "conversations": [
    {"from": "human", "value": "What is LoRA fine-tuning?"},
    {"from": "gpt", "value": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique..."}
  ]
}
```

- Multi-turn conversations with `from` and `value` fields
- Uses TRL's `DataCollatorForCompletionOnlyLM` for training

---

## How the Retry Loop Works

```
Iteration 1:  Dataset → Hyperparams → Train → Eval → Decision
                                                        │
                                          ┌─────────────┼──────────────┐
                                          │             │              │
                                       STOP ✓      CONTINUE → Train  ADJUST → Hyperparams
                                     (target met)  (more epochs)     (new config)
                                          │             │              │
                                       Report        Train          Train
                                                         │              │
                                                         └──────────────┘
                                                              ↓
                                                        Eval → Decision
                                                              ↓
                                                         (max 3 iterations)
```

**Decision logic:**
- **STOP**: Primary metric ≥ target value — training succeeded
- **CONTINUE**: Metric improving but needs more epochs (plateau reached, positive trend)
- **ADJUST**: Metric stagnant/declining — hyperparameters need to change
- **FAIL**: Max iterations reached, or training diverged (loss > 5.0 for 2 epochs)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload dataset file → returns `session_id`, file path |
| `POST` | `/api/start` | Start pipeline → begins SSE stream |
| `GET` | `/api/stream/{session_id}` | SSE event stream for live updates |
| `GET` | `/api/report/{session_id}` | Get final Markdown report |
| `GET` | `/api/report/{session_id}/json` | Get full JSON experiment report |
| `GET` | `/api/adapter/{session_id}` | Download adapter as .zip |
| `GET` | `/api/health` | Health check (GPU status, VRAM) |
| `GET` | `/api/models` | List supported base models |

### SSE Event Types

```jsonc
// Agent lifecycle
{ "type": "agent_start",    "agent": "dataset_agent",  "message": "Loading dataset..." }
{ "type": "agent_complete", "agent": "dataset_agent",  "data": { ... } }

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

## Switching LLM Providers

The orchestration LLM (which drives Hyperparameter and Decision agents) is **separate** from the model being fine-tuned. Set in `.env`:

```bash
# OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022

# Groq
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile

# Ollama (local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```

---

## Output Files

After a run completes, all artifacts are saved to the `outputs/` directory:

```
outputs/
├── adapters/{session_id}/          # LoRA adapter weights (HuggingFace-compatible)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── reports/{session_id}/
│   ├── report.json                 # Full experiment report (all metrics, hyperparams, logs)
│   └── report.md                   # Human-readable Markdown report
└── checkpoints/{session_id}/iter_0/  # Intermediate training checkpoints
    └── adapter/
```

**Adapter usage:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(model, "outputs/adapters/{session_id}")
```

---

## Frontend Dashboard

The dark-themed dashboard provides:

- **Upload Panel** — Drag-and-drop dataset upload with progress bar
- **Config Panel** — Model selection, template format, target metric, training toggles
- **Pipeline Visualizer** — 5 agent nodes with idle → active → complete state transitions
- **Live Loss Chart** — Chart.js line chart updating in real-time via SSE
- **Eval Score Cards** — BLEU, ROUGE-L, Perplexity, Exact Match with delta indicators
- **Hyperparameter Panel** — Current config with LLM reasoning
- **Agent Log** — Timestamped monospace log of all agent activity
- **Report Panel** — Markdown-rendered experiment report with download buttons

---

## Project Structure

```
lora_orchestrator/
├── backend/
│   ├── app/
│   │   ├── state.py              # OrchestrationState TypedDict (master state)
│   │   ├── graph.py              # LangGraph StateGraph with conditional routing
│   │   ├── llm.py                # LLM provider factory (OpenAI/Anthropic/Groq/Ollama)
│   │   ├── models.py             # Pydantic request/response models
│   │   ├── nodes/
│   │   │   ├── dataset_agent.py  # Dataset loading, cleaning, formatting
│   │   │   ├── hyperparam_agent.py # LLM-driven hyperparameter selection
│   │   │   ├── training_agent.py # LoRA fine-tuning with PEFT + TRL
│   │   │   ├── eval_agent.py     # BLEU/ROUGE/perplexity evaluation
│   │   │   ├── decision_agent.py # STOP/CONTINUE/ADJUST/FAIL decisions
│   │   │   └── report_agent.py   # Adapter export + report generation
│   │   └── utils/
│   │       ├── dataset_utils.py  # Dataset formatters, validators, splitters
│   │       ├── model_utils.py    # Model loading, quantization, LoRA config
│   │       ├── eval_utils.py     # Metric computation (BLEU, ROUGE, PPL)
│   │       ├── training_utils.py # Training arguments, epoch callback, early stop
│   │       └── report_utils.py   # JSON + Markdown report builder
│   ├── main.py                   # FastAPI app + SSE streaming
│   └── requirements.txt
├── frontend/
│   ├── index.html                # Main dashboard
│   ├── css/style.css             # Dark theme with CSS variables
│   └── js/
│       ├── app.js                # Global state + initialization
│       ├── upload.js             # Drag-drop file upload
│       ├── config.js             # Form config builder
│       ├── charts.js             # Chart.js loss visualization
│       ├── stream.js             # SSE event dispatcher
│       └── report.js             # Markdown report renderer
├── outputs/                      # Trained adapters, reports, checkpoints
├── uploads/                      # Uploaded datasets
└── .env.example                  # Environment configuration template
```

---

## Key Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `lora_r` (rank) | 4, 8, 16, 32, 64 | 16 | Higher rank = more capacity |
| `lora_alpha` | `lora_r × 2` | 32 | Scaling factor |
| `lora_dropout` | 0.0 – 0.3 | 0.05 | Higher for small datasets |
| `learning_rate` | 1e-5 – 5e-4 | 2e-4 | Lower for large models |
| `num_epochs` | 1 – 10 | 3 | Fewer for large datasets |
| `batch_size` | 1, 2, 4, 8 | 2 | VRAM-constrained |
| `lr_scheduler` | cosine, linear, constant | cosine | Cosine best for instruction tuning |
| `optimizer` | paged_adamw_8bit, adamw_torch | paged_adamw_8bit | Required with quantization |
| `max_seq_length` | 512, 1024, 2048 | 1024 | Depends on dataset |

---

## Constraints

- **Single GPU only** — no multi-GPU / distributed training (v1)
- **Max 2 hours** per training run before forced stop
- **Max 3 Decision Agent iterations** before escalating to human
- **4-bit quantization** default for 7B–14B models
- **No internet during training** — all downloads happen at dataset ingestion

---

## License

MIT
