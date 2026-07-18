# PRD.md — LoRA Fine-Tuning Orchestrator
> What we're building and why

---

## 1. Product Summary

A **LangGraph-based autonomous agent system** that orchestrates the entire LoRA
fine-tuning lifecycle end-to-end — from raw dataset ingestion to trained adapter
export — without requiring human intervention at each step.

The agent acts like a senior ML engineer: it curates the dataset, selects
hyperparameters, launches training, evaluates quality, decides whether to stop
early or adjust hyperparameters, and produces a final trained LoRA adapter
alongside a structured experiment report.

---

## 2. The Problem

Fine-tuning LLMs with LoRA today requires:

- Manual dataset cleaning and formatting into instruction-tuning templates
- Trial-and-error hyperparameter selection (rank, alpha, lr, epochs)
- Babysitting training runs — watching loss curves, catching divergence
- Manual evaluation after training with no standardized scoring
- Re-running experiments from scratch when results are poor
- Writing experiment reports manually after each run

This workflow is **slow, inconsistent, and expertise-gated**. A junior ML
engineer or researcher can spend 2-3 days on what should be a 2-3 hour process.

---

## 3. What We're Building

An autonomous orchestration pipeline with 5 specialist LangGraph agents:

| Agent | What it does |
|-------|-------------|
| **Dataset Agent** | Loads raw data, cleans it, formats into Alpaca/ShareGPT template, validates quality |
| **Hyperparameter Agent** | Selects LoRA rank, alpha, dropout, learning rate, scheduler based on dataset + base model |
| **Training Agent** | Launches fine-tuning via HuggingFace PEFT + TRL, streams loss logs back to state |
| **Evaluation Agent** | Runs BLEU, ROUGE, perplexity, and task-specific prompts on holdout set |
| **Decision Agent** | Analyzes metrics, decides: STOP (success) / CONTINUE (more epochs) / ADJUST (new hyperparams) / FAIL (escalate) |

A **FastAPI backend** exposes the pipeline over HTTP with SSE streaming so a
**dark-themed frontend dashboard** can show the training loop live — loss curves
updating in real-time, agent status, evaluation scores, and the final report.

---

## 4. Users

**Primary:** ML engineers and researchers who want to fine-tune LLMs on custom
datasets without hand-holding each step.

**Secondary:** Product teams who want to fine-tune a base model on company
data (support tickets, documentation, chat logs) and need a reproducible,
automated process.

---

## 5. Core Features (Must-Have for v1)

| # | Feature | Priority |
|---|---------|----------|
| 1 | Dataset ingestion: CSV, JSONL, HuggingFace dataset name | P0 |
| 2 | Auto-formatting to Alpaca and ShareGPT instruction templates | P0 |
| 3 | Dataset quality validation (length distribution, dedup, balance check) | P0 |
| 4 | LLM-driven hyperparameter selection with reasoning | P0 |
| 5 | LoRA fine-tuning via PEFT + TRL (SFTTrainer) | P0 |
| 6 | Real-time loss streaming to frontend via SSE | P0 |
| 7 | Automatic early stopping on loss plateau or divergence | P0 |
| 8 | Post-training evaluation: BLEU, ROUGE-L, perplexity | P0 |
| 9 | Decision loop: adjust hyperparams and re-train if below threshold | P0 |
| 10 | LoRA adapter export (HuggingFace-compatible) | P0 |
| 11 | Experiment report: JSON + Markdown summary | P0 |
| 12 | Frontend dashboard: live loss chart, agent pipeline, eval scores | P0 |
| 13 | Support base models: Qwen2.5, Llama-3, Mistral, Phi-3 | P1 |
| 14 | W&B / MLflow experiment tracking integration | P1 |
| 15 | Dataset preview and statistics panel in UI | P1 |

---

## 6. Out of Scope (v1)

- Full fine-tuning (non-LoRA) — LoRA only
- Multi-GPU / distributed training
- RLHF / DPO / reward modeling
- Model merging (LoRA adapter → full model)
- Deployment / inference serving after training
- User authentication or multi-tenancy
- Cloud training (AWS, GCP) — local GPU only
- Hyperparameter search (grid/random/Bayesian) — LLM selects once per iteration

---

## 7. Success Criteria

The product is done when:

1. A user can drop a raw JSONL dataset, pick a base model, and click "Start" —
   the system produces a trained LoRA adapter with no further input.
2. The frontend shows live loss curves updating every epoch.
3. The Decision Agent correctly triggers a retry with adjusted hyperparameters
   when eval score is below the threshold.
4. The final adapter loads and runs inference correctly with HuggingFace `peft`.
5. All 5 agents complete without error on Qwen2.5-7B and Llama-3-8B.
6. The experiment report is complete: dataset stats, hyperparams used, training
   curve, eval scores, decision log, adapter path.

---

## 8. Constraints

- **Hardware:** Must run on a single GPU (minimum 16GB VRAM, target 24GB)
- **Base models:** 7B-8B parameter models via 4-bit quantization (bitsandbytes)
- **Training time:** Max 2 hours per run before forced stop
- **Max retries:** 3 Decision Agent iterations before escalating to human
- **LLM for orchestration:** Any provider (OpenAI / Anthropic / Groq / Ollama)
  — orchestration LLM is separate from the model being fine-tuned
- **No internet during training:** All HuggingFace downloads happen at ingest,
  not during the training loop

---

## 9. Non-Goals

- We are NOT building a general ML platform (no arbitrary model support)
- We are NOT building a dataset annotation tool
- We are NOT building an inference API for the fine-tuned model
- We are NOT replacing W&B or MLflow — we integrate with them, not compete
