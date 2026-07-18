import os
import json
import uuid
import asyncio
import time
from typing import Dict
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from backend.app.models import UploadResponse, StartRequest, StartResponse, HealthResponse, ModelsResponse
from backend.app.graph import build_graph
from backend.app.llm import build_llm

load_dotenv()

SUPPORTED_MODELS = {
    "qwen2.5-7b": {"key": "qwen2.5-7b", "name": "Qwen2.5-7B-Instruct", "params": "7B", "vram_required_gb": 16},
    "qwen2.5-14b": {"key": "qwen2.5-14b", "name": "Qwen2.5-14B-Instruct", "params": "14B", "vram_required_gb": 24},
    "llama-3-8b": {"key": "llama-3-8b", "name": "Meta-Llama-3-8B-Instruct", "params": "8B", "vram_required_gb": 16},
    "mistral-7b": {"key": "mistral-7b", "name": "Mistral-7B-Instruct-v0.3", "params": "7B", "vram_required_gb": 16},
    "phi-3-mini": {"key": "phi-3-mini", "name": "Phi-3-mini-4k-instruct", "params": "3.8B", "vram_required_gb": 8},
    "gemma-2-9b": {"key": "gemma-2-9b", "name": "gemma-2-9b-it", "params": "9B", "vram_required_gb": 16},
}

MODEL_IDS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
}

sessions: Dict[str, dict] = {}
session_queues: Dict[str, asyncio.Queue] = {}
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs/adapters", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    try:
        llm = build_llm()
        graph = build_graph(llm)
    except Exception:
        graph = build_graph(None)
    yield


app = FastAPI(title="LoRA Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("outputs"):
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".csv", ".jsonl", ".json"}:
        raise HTTPException(400, "Only .csv, .jsonl, .json files accepted")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 500MB)")

    session_id = str(uuid.uuid4())
    save_path = os.path.join("uploads", f"{session_id}_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        session_id=session_id,
        file_path=save_path,
        file_name=file.filename,
        file_size_mb=round(len(content) / (1024 * 1024), 2),
    )


@app.post("/api/start", response_model=StartResponse)
async def start_pipeline(req: StartRequest):
    if req.session_id not in sessions and not os.path.exists(req.file_path):
        raise HTTPException(404, "Session not found. Upload a file first.")

    base_model_id = MODEL_IDS.get(req.base_model, "Qwen/Qwen2.5-7B-Instruct")

    initial_state = {
        "run_config": {
            "session_id": req.session_id,
            "dataset_path": req.file_path,
            "dataset_format": os.path.splitext(req.file_name)[1].lstrip("."),
            "dataset_name": req.file_name,
            "template_format": req.template_format,
            "base_model": req.base_model,
            "base_model_id": base_model_id,
            "target_metric": req.target_metric,
            "target_value": req.target_value,
            "max_iterations": req.max_iterations,
            "max_samples": req.max_samples,
            "quantization": req.quantization,
            "use_wandb": req.use_wandb,
            "use_mlflow": req.use_mlflow,
        },
        "dataset": {},
        "current_hyperparams": {},
        "hyperparam_reasoning": "",
        "training": {},
        "eval_results": {},
        "eval_history": [],
        "hyperparam_history": [],
        "decision_history": [],
        "training_logs": [],
        "agent_logs": [],
        "decision": "",
        "decision_reasoning": "",
        "suggested_hyperparam_changes": None,
        "iteration": 0,
        "report": {},
        "status": "initializing",
        "current_agent": "",
        "error": None,
    }

    sessions[req.session_id] = initial_state
    session_queues[req.session_id] = asyncio.Queue()

    asyncio.create_task(run_pipeline(req.session_id, initial_state))

    return StartResponse(
        session_id=req.session_id,
        stream_url=f"/api/stream/{req.session_id}",
        status="started",
    )


async def run_pipeline(session_id: str, state: dict):
    queue = session_queues[session_id]
    try:
        async for event in graph.astream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                sse_event = {
                    "type": "agent_start",
                    "agent": node_name,
                    "message": f"Running {node_name}...",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                await queue.put(sse_event)

                state.update(node_output)

                complete_event = {
                    "type": "agent_complete",
                    "agent": node_name,
                    "data": {k: v for k, v in node_output.items() if k != "agent_logs"},
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                await queue.put(complete_event)

                if "training" in node_output and isinstance(node_output["training"], dict):
                    for log in node_output["training"].get("logs", []):
                        log_event = {
                            "type": "training_log",
                            "epoch": log.get("epoch", 0),
                            "train_loss": log.get("train_loss", 0),
                            "val_loss": log.get("val_loss", 0),
                            "lr": log.get("learning_rate", 0),
                            "elapsed_sec": log.get("elapsed_sec", 0),
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        await queue.put(log_event)
                    stop_reason = node_output["training"].get("stop_reason")
                    if stop_reason and stop_reason in ("divergence", "plateau"):
                        stop_event = {
                            "type": "early_stop",
                            "reason": stop_reason,
                            "epoch": node_output["training"].get("total_epochs_run", 0),
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        await queue.put(stop_event)

                if "eval_results" in node_output and isinstance(node_output["eval_results"], dict):
                    er = node_output["eval_results"]
                    eval_event = {
                        "type": "eval_complete",
                        "bleu": er.get("bleu", 0),
                        "rouge_l": er.get("rouge_l", 0),
                        "perplexity": er.get("perplexity", 0),
                        "exact_match": er.get("exact_match", 0),
                        "primary_metric": er.get("primary_metric", ""),
                        "primary_value": er.get("primary_value", 0),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    await queue.put(eval_event)

                if "decision" in node_output:
                    dec_event = {
                        "type": "decision",
                        "decision": node_output.get("decision", ""),
                        "reasoning": node_output.get("decision_reasoning", ""),
                        "iteration": node_output.get("iteration", 0),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    await queue.put(dec_event)

        final_event = {
            "type": "complete",
            "adapter_path": state.get("report", {}).get("adapter_path", ""),
            "report_url": f"/api/report/{session_id}",
            "final_metric": state.get("eval_results", {}).get("primary_value", 0),
            "total_iterations": state.get("iteration", 0),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        await queue.put(final_event)

    except Exception as e:
        error_event = {
            "type": "error",
            "message": str(e),
            "agent": state.get("current_agent", "unknown"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        await queue.put(error_event)


@app.get("/api/stream/{session_id}")
async def stream_events(session_id: str):
    if session_id not in session_queues:
        raise HTTPException(404, "Session not found")

    queue = session_queues[session_id]

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
                yield {"data": json.dumps(event)}
                if event.get("type") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "ping", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})}

    return EventSourceResponse(event_generator())


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    report_path = os.path.join("outputs", "reports", session_id, "report.md")
    if not os.path.exists(report_path):
        raise HTTPException(404, "Report not found")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content, media_type="text/markdown")


@app.get("/api/report/{session_id}/json")
async def get_report_json(session_id: str):
    report_path = os.path.join("outputs", "reports", session_id, "report.json")
    if not os.path.exists(report_path):
        raise HTTPException(404, "Report not found")
    import json as json_mod
    with open(report_path, "r", encoding="utf-8") as f:
        content = json_mod.load(f)
    return content


@app.get("/api/adapter/{session_id}")
async def download_adapter(session_id: str):
    adapter_dir = os.path.join("outputs", "adapters", session_id)
    if not os.path.exists(adapter_dir):
        raise HTTPException(404, "Adapter not found")
    import zipfile
    import io
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(adapter_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, adapter_dir)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=adapter_{session_id}.zip"},
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available()
    vram_gb = None
    if gpu_available:
        vram_gb = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
    return HealthResponse(status="ok", gpu_available=gpu_available, vram_gb=vram_gb)


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(models=list(SUPPORTED_MODELS.values()))
