import json
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = """You are an ML experiment director. You analyze fine-tuning results and decide
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
    "learning_rate": float,
    "num_epochs": int
  } | null
}

Note: "stop" and "fail" decisions are handled in code — never return them here."""


def build_decision_message(state: dict) -> str:
    cfg = state.get("run_config", {})
    eval_r = state.get("eval_results", {})
    hist = state.get("eval_history", [])
    itr = state.get("iteration", 0)
    training = state.get("training", {})

    trend_block = ""
    if len(hist) >= 2:
        delta = hist[-1].get("primary_value", 0) - hist[-2].get("primary_value", 0)
        trend_block = f"""
TREND ANALYSIS:
- Previous {cfg.get('target_metric', 'rouge_l')}: {hist[-2].get('primary_value', 0):.4f}
- Current  {cfg.get('target_metric', 'rouge_l')}: {hist[-1].get('primary_value', 0):.4f}
- Delta: {delta:+.4f} ({'improving' if delta > 0 else 'declining' if delta < 0 else 'flat'})
"""

    return f"""
ITERATION: {itr + 1}
ITERATIONS REMAINING: {cfg.get('max_iterations', 3) - itr - 1}

CURRENT HYPERPARAMETERS:
{json.dumps(state.get('current_hyperparams', {}), indent=2)}

TRAINING OUTCOME:
- Epochs run: {training.get('total_epochs_run', 0)}
- Best loss: {training.get('best_loss', 0):.4f}
- Best epoch: {training.get('best_epoch', 0)}
- Stop reason: {training.get('stop_reason') or 'completed all epochs'}

EVALUATION RESULTS:
- BLEU:         {eval_r.get('bleu', 0):.4f}
- ROUGE-L:      {eval_r.get('rouge_l', 0):.4f}
- Perplexity:   {eval_r.get('perplexity', 0):.2f}
- Exact Match:  {eval_r.get('exact_match', 0):.4f}
- PRIMARY ({cfg.get('target_metric', 'rouge_l')}): {eval_r.get('primary_value', 0):.4f}
- TARGET:       {cfg.get('target_value', 0)}
- GAP:          {cfg.get('target_value', 0) - eval_r.get('primary_value', 0):+.4f}
{trend_block}
Decide: continue (more epochs) or adjust (new hyperparameters)?
"""


def decision_node(state: dict, llm=None) -> dict:
    cfg = state.get("run_config", {})
    eval_r = state.get("eval_results", {})
    itr = state.get("iteration", 0)
    max_iter = cfg.get("max_iterations", 3)
    training = state.get("training", {})
    primary_value = eval_r.get("primary_value", 0)
    target_value = cfg.get("target_value", 0.6)

    # Automatic STOP
    if primary_value >= target_value:
        record = {
            "iteration": itr + 1,
            "decision": "stop",
            "reasoning": f"Primary metric {primary_value:.4f} >= target {target_value}. Training succeeded.",
            "metric_at_decision": primary_value,
            "suggested_changes": None,
        }
        return {
            "decision": "stop",
            "decision_reasoning": record["reasoning"],
            "suggested_hyperparam_changes": None,
            "iteration": itr + 1,
            "decision_history": [record],
            "agent_logs": [f"[Decision] STOP — target met ({primary_value:.4f} >= {target_value})"],
        }

    # Automatic FAIL
    if itr >= max_iter:
        record = {
            "iteration": itr + 1,
            "decision": "fail",
            "reasoning": f"Max iterations ({max_iter}) reached.",
            "metric_at_decision": primary_value,
            "suggested_changes": None,
        }
        return {
            "decision": "fail",
            "decision_reasoning": record["reasoning"],
            "suggested_hyperparam_changes": None,
            "iteration": itr + 1,
            "decision_history": [record],
            "agent_logs": [f"[Decision] FAIL — max iterations reached"],
        }

    if training.get("stop_reason") == "divergence":
        record = {
            "iteration": itr + 1,
            "decision": "fail",
            "reasoning": "Training diverged (loss > 5.0 for 2 consecutive epochs).",
            "metric_at_decision": primary_value,
            "suggested_changes": None,
        }
        return {
            "decision": "fail",
            "decision_reasoning": record["reasoning"],
            "suggested_hyperparam_changes": None,
            "iteration": itr + 1,
            "decision_history": [record],
            "agent_logs": [f"[Decision] FAIL — divergence detected"],
        }

    # LLM-based decision
    if llm is not None:
        user_msg = build_decision_message(state)
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            decision = parsed.get("decision", "adjust")
            reasoning = parsed.get("reasoning", "LLM decision.")
            changes = parsed.get("suggested_changes")
        except Exception:
            decision = "adjust"
            reasoning = "Failed to parse LLM response; defaulting to adjust."
            changes = None
    else:
        decision = "adjust"
        reasoning = "No LLM available; defaulting to adjust."
        changes = None

    record = {
        "iteration": itr + 1,
        "decision": decision,
        "reasoning": reasoning,
        "metric_at_decision": primary_value,
        "suggested_changes": changes,
    }

    return {
        "decision": decision,
        "decision_reasoning": reasoning,
        "suggested_hyperparam_changes": changes,
        "iteration": itr + 1,
        "decision_history": [record],
        "agent_logs": [f"[Decision] {decision.upper()} — {reasoning[:100]}"],
    }
