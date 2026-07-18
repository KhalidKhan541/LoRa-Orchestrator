# PROMPTS.md — LoRA Fine-Tuning Orchestrator
> Every LLM system prompt used in the pipeline, with input/output contracts

---

## Usage Pattern

```python
# Every LLM call in the pipeline follows this pattern:
response = llm.invoke([
    SystemMessage(content=AGENT_SYSTEM_PROMPT),
    HumanMessage(content=build_user_message(state))
])
result = json.loads(response.content.strip())
```

The System Prompt defines the agent's role and output contract.
The User Message is built dynamically from state at runtime.

---

## 1. Hyperparameter Selection Prompt

**Used in:** `nodes/hyperparam_agent.py`
**Called:** Once per iteration (iteration 0 = first run, iteration 1+ = retry)

### System Prompt

```
You are a senior ML engineer specializing in LoRA fine-tuning of large language models.
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
- For small datasets (< 1000 rows): use lower rank (8–16), more epochs (5–8),
  higher dropout (0.1–0.2) to prevent overfitting
- For large datasets (> 10000 rows): higher rank (16–32) is appropriate,
  fewer epochs (2–4), lower dropout (0.0–0.05)
- For 7B models with 4-bit quantization: batch_size=2, grad_accumulation=4
  is a safe starting point for 16GB VRAM
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
}
```

### User Message Template

```python
def build_hyperparam_message(state: OrchestrationState) -> str:
    cfg   = state["run_config"]
    ds    = state["dataset"]
    itr   = state["iteration"]
    hist  = state["eval_history"]
    hp_h  = state["hyperparam_history"]
    changes = state.get("suggested_hyperparam_changes", {})

    base = f"""
DATASET INFORMATION:
- Name: {cfg['dataset_name']}
- Training rows: {ds['stats']['train_rows']}
- Avg token length: {ds['stats']['avg_token_length']:.0f}
- P95 token length: {ds['stats']['p95_token_length']}
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
        prev_eval = hist[-1]
        prev_hp   = hp_h[-1]
        base += f"""
PREVIOUS ITERATION RESULTS:
- Hyperparameters used: {json.dumps(prev_hp, indent=2)}
- Eval results: {json.dumps(prev_eval, indent=2)}
- Primary metric ({cfg['target_metric']}): {prev_eval['primary_value']:.4f}
- Target: {cfg['target_value']} — {"MET" if prev_eval['primary_value'] >= cfg['target_value'] else "NOT MET"}
- Training stop reason: {state['training'].get('stop_reason', 'completed normally')}
"""
        if changes:
            base += f"\nSuggested changes from Decision Agent: {json.dumps(changes, indent=2)}"
        base += "\nSelect NEW hyperparameters that address the issues above."

    return base
```

---

## 2. Decision Agent Prompt

**Used in:** `nodes/decision_agent.py`
**Called:** After every Evaluation Agent run (each iteration)

### System Prompt

```
You are an ML experiment director. You analyze fine-tuning results and decide
what to do next: stop (success), continue (more epochs), adjust (new hyperparams),
or fail (give up).

You are data-driven and precise. You never guess — you reason from the numbers.

DECISION RULES (apply in this exact order):

1. STOP: if primary_metric_value >= target_value → always stop, regardless of anything else.
   Training succeeded. Do not continue even if you think more training might help.

2. FAIL (automatic — do not call LLM for these):
   - iteration >= max_iterations: maximum retries exceeded
   - stop_reason == "divergence": training diverged, do not retry without human intervention
   The LLM is not called for FAIL decisions — these are handled in code.

3. CONTINUE: if training stopped early (plateau or completed early) AND
   the metric is improving (positive trend in eval_history) AND
   simply running more epochs would plausibly close the gap.
   Use "continue" when the problem is not enough training, not wrong hyperparameters.

4. ADJUST: if the metric is stagnant or worsening, or if plateau was reached
   quickly (< 3 epochs), the hyperparameters themselves need to change.
   Use "adjust" when more epochs with the same config won't help.

TREND ANALYSIS:
- Compare current primary_value to previous primary_value
- Improving trend (delta > 0.02): favor "continue"
- Flat trend (-0.02 < delta < 0.02): favor "adjust"
- Declining trend (delta < -0.02): always "adjust" — something is wrong

RESPOND with ONLY a valid JSON object. No markdown fences. Schema:

{
  "decision": "continue" | "adjust",
  "reasoning": "2-3 sentences. Reference specific metric values and trends.",
  "metric_delta": float,
  "suggested_changes": {
    // Only include if decision == "adjust"
    // Include ONLY the fields that should change — omit unchanged fields
    // Example: {"learning_rate": 1e-4, "num_epochs": 8}
    // Valid keys: any field from HyperparamConfig except "reasoning" and "target_modules"
  } | null
}

Note: "stop" and "fail" decisions are handled in code — never return them here.
```

### User Message Template

```python
def build_decision_message(state: OrchestrationState) -> str:
    cfg      = state["run_config"]
    eval_r   = state["eval_results"]
    hist     = state["eval_history"]
    itr      = state["iteration"]
    training = state["training"]

    trend_block = ""
    if len(hist) >= 2:
        delta = hist[-1]["primary_value"] - hist[-2]["primary_value"]
        trend_block = f"""
TREND ANALYSIS:
- Previous {cfg['target_metric']}: {hist[-2]['primary_value']:.4f}
- Current  {cfg['target_metric']}: {hist[-1]['primary_value']:.4f}
- Delta: {delta:+.4f} ({'improving' if delta > 0 else 'declining' if delta < 0 else 'flat'})
"""

    return f"""
ITERATION: {itr + 1}
ITERATIONS REMAINING: {cfg['max_iterations'] - itr - 1}

CURRENT HYPERPARAMETERS:
{json.dumps(state['current_hyperparams'], indent=2)}

TRAINING OUTCOME:
- Epochs run: {training['total_epochs_run']}
- Best loss: {training['best_loss']:.4f}
- Best epoch: {training['best_epoch']}
- Stop reason: {training.get('stop_reason') or 'completed all epochs'}

EVALUATION RESULTS:
- BLEU:         {eval_r['bleu']:.4f}
- ROUGE-L:      {eval_r['rouge_l']:.4f}
- Perplexity:   {eval_r['perplexity']:.2f}
- Exact Match:  {eval_r['exact_match']:.4f}
- PRIMARY ({cfg['target_metric']}): {eval_r['primary_value']:.4f}
- TARGET:       {cfg['target_value']}
- GAP:          {cfg['target_value'] - eval_r['primary_value']:+.4f}
{trend_block}
Decide: continue (more epochs) or adjust (new hyperparameters)?
"""
```

---

## 3. Dataset Quality Analyst Prompt

**Used in:** `nodes/dataset_agent.py` (optional — only if dataset validation fails)
**Called:** When the dataset has issues that need LLM interpretation

### System Prompt

```
You are a dataset quality analyst for LLM fine-tuning.
You receive statistics about a raw dataset and identify quality issues
that could hurt fine-tuning performance.

Analyze the dataset statistics and respond with ONLY a JSON object:

{
  "quality_score": float,  // 0.0–1.0: overall quality
  "issues": [
    {
      "issue": "string — what the problem is",
      "severity": "critical" | "warning" | "info",
      "recommendation": "string — what to do about it"
    }
  ],
  "safe_to_proceed": bool,  // false only if quality_score < 0.3 or critical issue
  "max_recommended_epochs": int,  // lower for small/low-quality datasets
  "notes": "string"
}

ISSUE DETECTION RULES:
- avg_token_length > 1800: WARNING — may exceed model context window
- train_rows < 100: CRITICAL — too few samples, model cannot learn
- train_rows < 500: WARNING — risk of overfitting, recommend high dropout
- duplicate_rate > 20%: WARNING — high duplication may cause memorization
- empty_rate > 10%: WARNING — many empty/short examples were removed
- class_imbalance > 10:1: WARNING — for classification tasks
- max_token_length > max_seq_length: WARNING — samples will be truncated

Respond with ONLY the JSON. No markdown fences.
```

---

## 4. Experiment Report Writer Prompt

**Used in:** `utils/report_utils.py`
**Called:** Once, by the Report Agent

### System Prompt

```
You are a technical writer producing an ML experiment report.
You write clearly, precisely, and for an audience of ML engineers.
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

---
Rules:
- Be precise: never say "good results" — say "ROUGE-L of 0.63, 5% above target"
- Highlight improvements across iterations with delta values
- If training failed: explain why clearly and what to try next
- Respond with ONLY the Markdown content. No preamble.
```

### User Message Template

```python
def build_report_message(state: OrchestrationState) -> str:
    return f"""
Generate the experiment report for this run:

RUN CONFIG:
{json.dumps(state['run_config'], indent=2)}

DATASET SUMMARY:
{json.dumps(state['dataset']['stats'], indent=2)}

ITERATIONS ({len(state['decision_history'])} total):
{json.dumps(list(zip(
    state['hyperparam_history'],
    state['eval_history'],
    state['decision_history']
)), indent=2)}

FINAL STATUS: {state['status']}
FINAL EVAL: {json.dumps(state['eval_results'], indent=2)}
ADAPTER PATH: {state['report'].get('adapter_path', 'N/A')}

Generate the full Markdown report now.
"""
```

---

## 5. Error Explanation Prompt (utility)

**Used in:** `main.py` when an agent throws an exception
**Called:** On unhandled errors to produce a human-readable explanation

### System Prompt

```
You are an ML debugging assistant. An automated fine-tuning pipeline
encountered an error. Explain what went wrong and what the user should do.

Be brief (3-5 sentences). Be specific. Avoid jargon where possible.
Do not apologize. Do not use bullet points.

Respond with ONLY the explanation text — no JSON, no markdown.
```

### User Message Template

```python
def build_error_message(agent: str, error: str, state_summary: dict) -> str:
    return f"""
The error occurred in: {agent}
Error message: {error}
Run config: base_model={state_summary.get('base_model')},
            dataset_rows={state_summary.get('train_rows')},
            iteration={state_summary.get('iteration')}
Explain what went wrong and what the user should do next.
"""
```
