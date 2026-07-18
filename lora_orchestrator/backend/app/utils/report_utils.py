import json
import os
from datetime import datetime, timezone
from typing import Any


def build_json_report(state: dict) -> dict:
    cfg = state["run_config"]
    ds = state.get("dataset", {})
    stats = ds.get("stats", {})
    training = state.get("training", {})
    eval_results = state.get("eval_results", {})
    report = state.get("report", {})

    iterations = []
    for i, (hp, ev, dec) in enumerate(zip(
        state.get("hyperparam_history", []),
        state.get("eval_history", []),
        state.get("decision_history", []),
    )):
        iterations.append({
            "iteration": i + 1,
            "hyperparams": hp,
            "training": {
                "epochs_run": training.get("total_epochs_run", 0),
                "best_loss": training.get("best_loss", 0.0),
                "stop_reason": training.get("stop_reason"),
                "training_time_sec": training.get("training_time_sec", 0.0),
                "loss_curve": training.get("logs", []),
            },
            "eval_results": ev,
            "decision": dec,
        })

    best_idx = 0
    if state.get("eval_history"):
        best_idx = max(range(len(state["eval_history"])), key=lambda i: state["eval_history"][i].get("primary_value", 0))

    return {
        "session_id": cfg.get("session_id", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": state.get("status", "unknown"),
        "run_config": cfg,
        "dataset_summary": {
            "name": cfg.get("dataset_name", ""),
            "total_rows": stats.get("total_rows", 0),
            "train_rows": stats.get("train_rows", 0),
            "holdout_rows": stats.get("holdout_rows", 0),
            "avg_token_length": stats.get("avg_token_length", 0),
            "template_used": cfg.get("template_format", ""),
        },
        "iterations": iterations,
        "final_results": {
            "best_iteration": best_idx + 1,
            "best_hyperparams": state["hyperparam_history"][best_idx] if state.get("hyperparam_history") else {},
            "final_eval": eval_results,
            "adapter_path": report.get("adapter_path", ""),
            "total_training_time_sec": training.get("training_time_sec", 0),
        },
        "agent_logs": state.get("agent_logs", []),
    }


def build_markdown_report(state: dict, llm=None) -> str:
    if llm is None:
        return _build_fallback_markdown(state)
    from langchain_core.messages import SystemMessage, HumanMessage
    system_prompt = """You are a technical writer producing an ML experiment report.
Write clearly, precisely, and for an audience of ML engineers.
Every claim is backed by a specific number from the experiment data.

Write the report in Markdown with these exact sections:

# LoRA Fine-Tuning Experiment Report

## Executive Summary
2-3 sentences. State: base model, dataset name, final metric value vs target,
number of iterations, and whether the target was met.

## Configuration
Table of: base model, template format, target metric, target value, max iterations.

## Dataset Summary
- Rows: {train_rows} train / {holdout_rows} holdout
- Avg token length: X
- Template: Alpaca/ShareGPT
- Quality notes if any

## Iterations

### Iteration {N}
For each iteration:
- **Hyperparameters:** rank, alpha, lr, epochs, scheduler (table format)
- **Training:** epochs run, best loss, stop reason if early stopped
- **Evaluation:** table of BLEU / ROUGE-L / Perplexity / Exact Match
- **Decision:** what was decided and why (1-2 sentences)

## Final Results
- Best iteration: N
- Final {metric}: {value} (target was {target}) — MET ✓ / NOT MET ✗
- Total training time: X minutes
- Adapter saved to: {path}

## Recommendations
3-5 actionable suggestions for future runs based on what worked and what didn't.

## Appendix: Full Loss Curves
Table of epoch / train_loss / val_loss for the best iteration.

Rules:
- Be precise: never say "good results" — say "ROUGE-L of 0.63, 5% above target"
- Highlight improvements across iterations with delta values
- If training failed: explain why clearly and what to try next
- Respond with ONLY the Markdown content. No preamble."""

    user_msg = f"""Generate the experiment report for this run:

RUN CONFIG:
{json.dumps(state.get('run_config', {}), indent=2)}

DATASET SUMMARY:
{json.dumps(state.get('dataset', {}).get('stats', {}), indent=2)}

ITERATIONS ({len(state.get('decision_history', []))} total):
{json.dumps(list(zip(
    state.get('hyperparam_history', []),
    state.get('eval_history', []),
    state.get('decision_history', [])
)), indent=2)}

FINAL STATUS: {state.get('status', 'unknown')}
FINAL EVAL: {json.dumps(state.get('eval_results', {}), indent=2)}
ADAPTER PATH: {state.get('report', {}).get('adapter_path', 'N/A')}

Generate the full Markdown report now."""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
    return response.content


def _build_fallback_markdown(state: dict) -> str:
    cfg = state.get("run_config", {})
    eval_r = state.get("eval_results", {})
    training = state.get("training", {})
    ds = state.get("dataset", {}).get("stats", {})

    md = "# LoRA Fine-Tuning Experiment Report\n\n"
    md += "## Executive Summary\n\n"
    md += f"Base model: {cfg.get('base_model', 'N/A')}. "
    md += f"Dataset: {cfg.get('dataset_name', 'N/A')}. "
    md += f"Final {cfg.get('target_metric', 'N/A')}: {eval_r.get('primary_value', 0):.4f} "
    md += f"(target: {cfg.get('target_value', 0)}).\n\n"
    md += "## Dataset Summary\n\n"
    md += f"- Training rows: {ds.get('train_rows', 0)}\n"
    md += f"- Holdout rows: {ds.get('holdout_rows', 0)}\n"
    md += f"- Avg token length: {ds.get('avg_token_length', 0):.0f}\n\n"
    md += "## Evaluation Results\n\n"
    md += f"- BLEU: {eval_r.get('bleu', 0):.4f}\n"
    md += f"- ROUGE-L: {eval_r.get('rouge_l', 0):.4f}\n"
    md += f"- Perplexity: {eval_r.get('perplexity', 0):.2f}\n"
    md += f"- Exact Match: {eval_r.get('exact_match', 0):.4f}\n\n"
    md += "## Training\n\n"
    md += f"- Epochs run: {training.get('total_epochs_run', 0)}\n"
    md += f"- Best loss: {training.get('best_loss', 0):.4f}\n"
    md += f"- Stop reason: {training.get('stop_reason', 'completed')}\n"
    return md


def save_reports(session_id: str, json_report: dict, md_report: str, output_dir: str) -> None:
    report_dir = os.path.join(output_dir, "reports", session_id)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=str)
    with open(os.path.join(report_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md_report)
