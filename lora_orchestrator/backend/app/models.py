from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    session_id: str
    file_path: str
    file_name: str
    file_size_mb: float


class StartRequest(BaseModel):
    session_id: str
    file_path: str
    file_name: str
    template_format: str = "alpaca"
    base_model: str = "qwen2.5-7b"
    target_metric: str = "rouge_l"
    target_value: float = Field(0.60, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=5)
    max_samples: int = Field(10000, ge=100, le=500000)
    quantization: bool = True
    use_wandb: bool = False
    use_mlflow: bool = False
    llm_provider: str = "openai"
    llm_model: Optional[str] = None


class StartResponse(BaseModel):
    session_id: str
    stream_url: str
    status: str


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    vram_gb: Optional[float] = None


class ModelsResponse(BaseModel):
    models: list[dict]
