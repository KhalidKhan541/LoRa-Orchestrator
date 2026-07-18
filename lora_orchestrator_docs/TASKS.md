# TASKS.md — LoRA Fine-Tuning Orchestrator
> Ordered build checklist — implement in this exact order

---

## How to Use This File

Pass this file to your coding agent with this instruction:

```
Read PRD.md, SPEC.md, UI_SPEC.md, SCHEMA.md, and PROMPTS.md completely first.
Then implement every task below in the exact numbered order.
After completing each task, mark it [x].
Do not start the next task until the current one compiles/runs without errors.
Write all files to disk. Never output file contents as code blocks only.
```

---

## Phase 0 — Project Scaffold

- [ ] **0.1** Create the full directory structure from SPEC.md Section 2 exactly.
      Create all empty `__init__.py` files. Create empty placeholder files for every
      `.py`, `.js`, `.css`, `.html`, `.md` file listed. Create `uploads/`, `outputs/adapters/`,
      `outputs/reports/`, `outputs/checkpoints/` directories.

- [ ] **0.2** Create `backend/requirements.txt` with the exact packages from SPEC.md Section 8.

- [ ] **0.3** Create `.env.example`:
      ```
      # Orchestration LLM (NOT the model being fine-tuned)
      OPENAI_API_KEY=sk-...
      ANTHROPIC_API_KEY=sk-ant-...
      GROQ_API_KEY=gsk_...
      LLM_PROVIDER=openai
      LLM_MODEL=gpt-4o-mini

      # Paths
      UPLOAD_DIR=./uploads
      OUTPUT_DIR=./outputs

      # Training limits
      MAX_TRAINING_HOURS=2
      CODE_EXEC_TIMEOUT=7200

      # Optional tracking
      WANDB_API_KEY=
      MLFLOW_TRACKING_URI=
      ```

---

## Phase 1 — State and Core Utilities

- [ ] **1.1** Implement `backend/app/state.py` — copy the complete `OrchestrationState`
      TypedDict from SCHEMA.md Section 1 exactly. Include all nested TypedDicts:
      `RunConfig`, `DatasetStats`, `DatasetState`, `HyperparamConfig`, `EpochLog`,
      `TrainingState`, `EvalResults`, `DecisionRecord`, `ReportState`.

- [ ] **1.2** Implement `backend/app/llm.py` — provider factory supporting:
      `openai`, `anthropic`, `groq`, `ollama`. Default temperature `0.2` for all.
      Read provider and model from env vars with fallback to function args.

- [ ] **1.3** Implement `backend/app/utils/dataset_utils.py`:
      - `load_dataset(path: str, format: str, max_samples: int) -> pd.DataFrame`
        Supports: "csv" (pandas), "jsonl" (json.loads per line), "huggingface" (datasets lib)
      - `deduplicate(df: pd.DataFrame, col: str) -> pd.DataFrame`
      - `filter_empty(df: pd.DataFrame, min_chars: int = 10) -> pd.DataFrame`
      - `format_alpaca(row: dict) -> dict` — returns `{"text": "..."}`
        Use template from SCHEMA.md Section 3
      - `format_sharegpt(row: dict) -> dict` — returns `{"conversations": [...]}`
      - `compute_stats(df: pd.DataFrame, tokenizer) -> DatasetStats`
        Compute: total_rows, train_rows, holdout_rows, duplicates_removed, empty_removed,
        avg_token_length, max_token_length, min_token_length, p95_token_length
      - `split_dataset(df, holdout_ratio=0.1) -> tuple[pd.DataFrame, pd.DataFrame]`
      - `save_jsonl(df: pd.DataFrame, path: str) -> None`

- [ ] **1.4** Implement `backend/app/utils/model_utils.py`:
      - `load_model(model_id: str, quantization: bool) -> tuple[model, tokenizer]`
        4-bit config: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`
      - `apply_lora(model, hyperparams: HyperparamConfig, model_family: str) -> model`
        Use `TARGET_MODULES` dict from SPEC.md Section 7 to look up target modules by model family
      - `save_adapter(model, path: str) -> str` — saves with `model.save_pretrained(path)`
      - `load_adapter(base_model_id: str, adapter_path: str, quantization: bool)`
        Returns model+tokenizer for evaluation

- [ ] **1.5** Implement `backend/app/utils/eval_utils.py`:
      - `compute_bleu(predictions: list[str], references: list[str]) -> float`
        Use `nltk.translate.bleu_score.corpus_bleu`
      - `compute_rouge_l(predictions: list[str], references: list[str]) -> float`
        Use `rouge_score.rouge_scorer.RougeScorer(['rougeL'])`
      - `compute_perplexity(model, tokenizer, texts: list[str]) -> float`
        Mean per-token perplexity: exp(mean cross-entropy loss)
      - `compute_exact_match(predictions: list[str], references: list[str]) -> float`
        Case-insensitive strip comparison
      - `run_inference(model, tokenizer, prompts: list[str], batch_size=4) -> list[str]`
        Generate with `max_new_tokens=256`, `do_sample=False`

- [ ] **1.6** Implement `backend/app/utils/training_utils.py`:
      - `build_training_args(hyperparams: HyperparamConfig, output_dir: str) -> TrainingArguments`
        Map all hyperparams to HuggingFace `TrainingArguments` fields
      - `EpochLogCallback(TrainerCallback)` — custom callback:
        Override `on_epoch_end`: append `EpochLog` dict to `self.epoch_logs` list,
        also call `self.sse_callback(epoch_log)` if provided
      - `check_early_stop(logs: List[EpochLog]) -> Optional[str]`
        Implement exactly from SCHEMA.md Section 7

- [ ] **1.7** Implement `backend/app/utils/report_utils.py`:
      - `build_json_report(state: OrchestrationState) -> dict`
        Serialize state into `ExperimentReport` schema from SCHEMA.md Section 5
      - `build_markdown_report(state: OrchestrationState, llm) -> str`
        Call LLM with PROMPTS.md Section 4 prompt
      - `save_reports(session_id: str, json_report: dict, md_report: str, output_dir: str)`
        Save to `outputs/reports/{session_id}/report.json` and `report.md`

---

## Phase 2 — Agent Nodes

- [ ] **2.1** Implement `backend/app/nodes/dataset_agent.py`:
      - Main function: `dataset_node(state: OrchestrationState, llm) -> dict`
      - Load dataset using `dataset_utils.load_dataset()`
      - Deduplicate and filter
      - Format into template (alpaca or sharegpt)
      - Compute stats using a fast tokenizer (`AutoTokenizer.from_pretrained` with `use_fast=True`)
        NOTE: use tokenizer for the configured base model
      - Split 90/10 train/holdout
      - Save both splits to `uploads/{session_id}_train.jsonl` and `_holdout.jsonl`
      - Return state update dict (see SPEC.md Section 4.1 for exact keys)
      - Log entry format: `f"[Dataset] Processed {n} rows → {train} train / {holdout} holdout"`

- [ ] **2.2** Implement `backend/app/nodes/hyperparam_agent.py`:
      - Main function: `hyperparam_node(state: OrchestrationState, llm) -> dict`
      - Build user message using `build_hyperparam_message()` from PROMPTS.md Section 1
      - Call LLM with system + user message
      - Parse JSON response — strip markdown fences if present
      - Validate all fields against constraints in SCHEMA.md Section 6
        On validation failure: apply safe defaults and log a warning
      - Look up `target_modules` from `TARGET_MODULES` dict using `base_model` key
        (override whatever LLM returned — always use the dict value)
      - Return state update dict (see SPEC.md Section 4.2)
      - Also apply `suggested_hyperparam_changes` from state if iteration > 0 and
        Decision Agent provided them — merge them into the LLM output before validation

- [ ] **2.3** Implement `backend/app/nodes/training_agent.py`:
      - Main function: `training_node(state: OrchestrationState) -> dict`
      - Load base model + tokenizer via `model_utils.load_model()`
      - Apply LoRA via `model_utils.apply_lora()`
      - Build `SFTTrainer` with:
        - dataset from `state["dataset"]["train_path"]`
        - `EpochLogCallback` with an `sse_queue` (use `asyncio.Queue` or a list)
        - `EarlyStoppingCallback` — but implement custom logic via `check_early_stop()`
          in `on_epoch_end` rather than HuggingFace's built-in (it uses eval_loss differently)
      - Run `trainer.train()`
      - After training: check `check_early_stop()` on full log list
      - Save checkpoint to `outputs/checkpoints/{session_id}/iter_{iteration}/`
      - Return state update dict (see SPEC.md Section 4.3)
      - IMPORTANT: release GPU memory after training:
        ```python
        del model
        torch.cuda.empty_cache()
        gc.collect()
        ```

- [ ] **2.4** Implement `backend/app/nodes/eval_agent.py`:
      - Main function: `eval_node(state: OrchestrationState) -> dict`
      - Load adapter from checkpoint via `model_utils.load_adapter()`
      - Load holdout dataset from `state["dataset"]["holdout_path"]`
      - Extract prompts (instruction + input) and references (output) from holdout
      - Run `eval_utils.run_inference()` to get predictions
      - Compute all 4 metrics: BLEU, ROUGE-L, perplexity, exact_match
      - Set `primary_value` based on `run_config.target_metric`
      - Release GPU memory after eval (same pattern as training_node)
      - Return state update dict (see SPEC.md Section 4.4)

- [ ] **2.5** Implement `backend/app/nodes/decision_agent.py`:
      - Main function: `decision_node(state: OrchestrationState, llm) -> dict`
      - Check automatic STOP condition first (in code, no LLM):
        `if primary_value >= target_value → return decision="stop"`
      - Check automatic FAIL conditions (in code, no LLM):
        `if iteration >= max_iterations → decision="fail"`
        `if stop_reason == "divergence" → decision="fail"`
      - Only call LLM if none of the above matched
      - Build user message using `build_decision_message()` from PROMPTS.md Section 2
      - Parse LLM response → extract "decision" and "reasoning"
      - Increment `iteration` counter
      - Build `DecisionRecord` and append to `decision_history`
      - Return state update dict (see SPEC.md Section 4.5)

- [ ] **2.6** Implement `backend/app/nodes/report_agent.py`:
      - Main function: `report_node(state: OrchestrationState, llm) -> dict`
      - Export final adapter: find best iteration checkpoint, copy to
        `outputs/adapters/{session_id}/`
      - Build JSON report via `report_utils.build_json_report()`
      - Build Markdown report via `report_utils.build_markdown_report()` (calls LLM)
      - Save both via `report_utils.save_reports()`
      - Set `status = "done"` if `state["decision"] == "stop"`, else `"failed"`
      - Return state update dict (see SPEC.md Section 4.6)

---

## Phase 3 — Graph Assembly

- [ ] **3.1** Implement `backend/app/graph.py`:
      - Implement `route_decision(state) -> str` from SPEC.md Section 6
      - Assemble `StateGraph` with all 6 nodes from SPEC.md Section 3
      - Add all edges and conditional edges exactly as shown
      - `build_graph(llm) -> CompiledGraph` factory function
      - The compiled graph is the return value — `main.py` calls `build_graph(llm)` once at startup

---

## Phase 4 — FastAPI Backend

- [ ] **4.1** Implement `backend/app/models.py` — all Pydantic models from SCHEMA.md Section 2:
      `UploadResponse`, `StartRequest`, `StartResponse`, `HealthResponse`, `ModelsResponse`

- [ ] **4.2** Implement `backend/main.py`:
      - FastAPI app with CORS for all origins
      - On startup: create `uploads/`, `outputs/adapters/`, `outputs/reports/`,
        `outputs/checkpoints/` if they don't exist
      - Mount `outputs/` as `/outputs` static directory
      - Load env vars with `python-dotenv`
      - Build LLM and compile graph once at startup (`lifespan` context manager)
      - Store active sessions in a dict: `sessions: Dict[str, OrchestrationState] = {}`

- [ ] **4.3** Implement `POST /api/upload`:
      - Accept multipart file upload
      - Validate extension: `.csv`, `.jsonl`, `.json` only → 400 otherwise
      - Validate size ≤ 500MB → 400 otherwise
      - Generate `session_id = str(uuid.uuid4())`
      - Save to `uploads/{session_id}_{filename}`
      - Return `UploadResponse`

- [ ] **4.4** Implement `POST /api/start`:
      - Validate `session_id` exists (file was uploaded) → 404 otherwise
      - Build initial `OrchestrationState` from `StartRequest`
      - Start pipeline in background: `asyncio.create_task(run_pipeline(session_id, state))`
      - Return `StartResponse` immediately (don't wait for pipeline to finish)
      - `run_pipeline()` is an async function that calls `graph.astream(state)` and
        pushes SSE events into a per-session `asyncio.Queue`

- [ ] **4.5** Implement `GET /api/stream/{session_id}` (SSE endpoint):
      - Use `sse_starlette.sse.EventSourceResponse`
      - Generator reads from the session's `asyncio.Queue`
      - Yields `ServerSentEvent(data=json.dumps(event))` for each event
      - All SSE event schemas are in SCHEMA.md Section 4 — use them exactly
      - Closes when `type == "complete"` or `type == "error"` event is sent
      - Add `ping` every 15 seconds to keep connection alive

- [ ] **4.6** Implement remaining endpoints:
      - `GET /api/report/{session_id}` → read `outputs/reports/{session_id}/report.md`,
        return as `text/markdown`
      - `GET /api/report/{session_id}/json` → return `report.json` as JSON
      - `GET /api/adapter/{session_id}` → zip `outputs/adapters/{session_id}/`,
        return as `application/zip` download with filename `adapter_{session_id}.zip`
      - `GET /api/health` → check `torch.cuda.is_available()`, return `HealthResponse`
      - `GET /api/models` → return list of supported models from SPEC.md Section 7

---

## Phase 5 — Frontend

- [ ] **5.1** Implement `frontend/css/style.css`:
      - CSS variables: copy every token from UI_SPEC.md Section 2 exactly
      - Base reset: `*, *::before, *::after { box-sizing: border-box; }`
      - Body: `background: var(--bg-base); color: var(--text-primary); font-family: var(--font-sans);`
      - Two-panel layout: CSS Grid `grid-template-columns: 380px 1fr`
      - All panel styles: `--bg-panel` background, `--border-subtle` border,
        `--radius-md` border-radius, `--shadow-panel` box-shadow
      - Agent node styles: `.node-idle`, `.node-active`, `.node-complete`, `.node-error`
        each with correct border-color from UI_SPEC.md Section 3.4
      - Spinning animation for `.node-active`: `@keyframes spin` on the ring element
      - Progress bar animation: `@keyframes fill` left to right
      - Drag-over state: `.upload-zone.drag-over` with `--accent-dataset` border
      - Metric card styles: `--bg-card` background, centered content
      - Log panel: `font-family: var(--font-mono)`, `font-size: 12px`, `overflow-y: auto`
      - All toggle switches: pill shape, transitions

- [ ] **5.2** Implement `frontend/index.html`:
      - `<!DOCTYPE html>` with dark theme meta tags
      - Import CDN scripts from UI_SPEC.md Section 7 (Chart.js + marked.js)
      - Import all 6 JS files and `style.css`
      - Header with title + status badge + session ID display
      - Left panel: upload zone, config form (all fields from UI_SPEC.md Section 3.2),
        start button, agent log panel
      - Right panel: pipeline visualizer (5 nodes + arrows as SVG or CSS flexbox),
        loss chart canvas `<canvas id="lossChart">`, eval scores grid (4 cards),
        hyperparameter log panel, report panel (hidden initially)
      - All element IDs must be stable — JS files reference them by ID

- [ ] **5.3** Implement `frontend/js/upload.js`:
      - `initUpload()` — attach drag-drop and click-to-browse to `#uploadZone`
      - `handleDrop(files)` — validate type/size, call `uploadFile()`
      - `uploadFile(file)` — `fetch POST /api/upload` with FormData, track progress,
        on success: store `sessionId` in `window.app.sessionId`, show filename
      - Export: `{ initUpload }`

- [ ] **5.4** Implement `frontend/js/config.js`:
      - `getConfig()` — read all form fields, return `StartRequest` payload object
      - `populateModelSelect()` — fetch `GET /api/models`, populate `#baseModelSelect`
      - `setFormEnabled(bool)` — disable/enable all inputs during run
      - Export: `{ getConfig, populateModelSelect, setFormEnabled }`

- [ ] **5.5** Implement `frontend/js/charts.js`:
      - `initChart()` — initialize Chart.js on `#lossChart` with two datasets:
        train_loss (amber) and val_loss (blue). Config from UI_SPEC.md Section 3.5.
      - `addEpochPoint(epoch, trainLoss, valLoss)` — push data point, call `chart.update()`
      - `addEarlyStopMarker(epoch, reason)` — add vertical red dashed annotation line
      - `resetChart()` — clear all data for new run
      - Export: `{ initChart, addEpochPoint, addEarlyStopMarker, resetChart }`

- [ ] **5.6** Implement `frontend/js/stream.js`:
      - `startStream(sessionId)` — open `EventSource` on `/api/stream/{sessionId}`
      - Dispatch table for each SSE event type from SCHEMA.md Section 4:
        - `agent_start` → update node to "active" state, log entry
        - `agent_complete` → update node to "complete" state, update HP panel if hyperparam_agent
        - `training_log` → `charts.addEpochPoint()`, log entry
        - `early_stop` → `charts.addEarlyStopMarker()`, log entry, warning style
        - `eval_complete` → update 4 metric cards with values + delta
        - `decision` → pulse decision node, log entry with decision color
        - `complete` → update all nodes to complete, call `report.showReport()`
        - `error` → update current node to error, log entry in red, show error panel
      - `setNodeState(agent, state)` — toggle `.node-idle/active/complete/error` classes
      - `addLogEntry(agent, message)` — prepend timestamped entry to log panel
      - `stopStream()` — close EventSource
      - Export: `{ startStream, stopStream }`

- [ ] **5.7** Implement `frontend/js/report.js`:
      - `showReport(reportUrl, adapterSessionId)` — reveal `#reportPanel`
      - Fetch report Markdown from `reportUrl`
      - Render to HTML via `marked.parse()`
      - Inject into `#reportContent`
      - Wire `#downloadReportBtn` → fetch report, trigger download as `.md`
      - Wire `#downloadAdapterBtn` → navigate to `/api/adapter/{sessionId}`
      - Export: `{ showReport }`

- [ ] **5.8** Implement `frontend/js/app.js`:
      - `window.app = { sessionId: null, isRunning: false }`
      - On DOMContentLoaded:
        - Call `upload.initUpload()`
        - Call `config.populateModelSelect()`
        - Call `charts.initChart()`
        - Attach `#startBtn` click handler
      - Start button handler:
        - Validate `window.app.sessionId` exists → alert if not
        - Get config via `config.getConfig()`
        - `POST /api/start` with payload
        - On success: disable form, change button to "Stop Run", call `stream.startStream()`
      - Stop button handler:
        - Call `stream.stopStream()`
        - Re-enable form, reset button

---

## Phase 6 — Integration and README

- [ ] **6.1** Write `README.md`:
      - One-paragraph description
      - Prerequisites: Python 3.11+, CUDA GPU (16GB+ VRAM), Node.js not required
      - Quickstart (5 commands):
        ```bash
        pip install -r backend/requirements.txt
        cp .env.example .env
        # Edit .env — add OPENAI_API_KEY (or your chosen provider)
        uvicorn backend.main:app --reload --port 8000
        # Open frontend/index.html in browser
        ```
      - Supported base models table
      - Dataset format guide (Alpaca and ShareGPT examples)
      - How to switch LLM providers
      - How the retry loop works (with diagram)
      - Output files explanation

- [ ] **6.2** End-to-end smoke test:
      - Prepare a minimal 200-row Alpaca-format JSONL dataset
      - Upload via the frontend, configure Qwen2.5-7B, target ROUGE-L 0.4,
        max_iterations=1
      - Verify: all 5 agent nodes complete (dataset may use mock if no GPU),
        loss chart shows data, eval scores appear, report renders
      - Fix any integration issues

---

## Definition of Done

All tasks marked [x] AND:

- [ ] `uvicorn backend.main:app` starts without import errors
- [ ] `POST /api/upload` accepts a JSONL file and returns `session_id`
- [ ] `POST /api/start` returns immediately (non-blocking)
- [ ] `GET /api/stream/{id}` streams SSE events in correct format
- [ ] Frontend pipeline nodes transition idle→active→complete during a run
- [ ] Loss chart updates live during training
- [ ] Report panel renders Markdown after pipeline completes
- [ ] Adapter zip downloads successfully
- [ ] No hardcoded API keys anywhere in the codebase
- [ ] `.env.example` covers all required variables
