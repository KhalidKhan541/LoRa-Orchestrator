from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator


class RunConfig(TypedDict):
    session_id: str
    dataset_path: str
    dataset_format: str
    dataset_name: str
    template_format: str
    base_model: str
    base_model_id: str
    target_metric: str
    target_value: float
    max_iterations: int
    max_samples: int
    quantization: bool
    use_wandb: bool
    use_mlflow: bool


class DatasetStats(TypedDict):
    total_rows: int
    train_rows: int
    holdout_rows: int
    duplicates_removed: int
    empty_removed: int
    avg_token_length: float
    max_token_length: int
    min_token_length: int
    p95_token_length: int
    column_names: List[str]
    template_format: str


class DatasetState(TypedDict):
    stats: DatasetStats
    train_path: str
    holdout_path: str
    sample_count: int
    avg_token_length: float
    ready: bool


class HyperparamConfig(TypedDict):
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    bias: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    grad_accumulation: int
    warmup_ratio: float
    lr_scheduler: str
    optimizer: str
    weight_decay: float
    max_grad_norm: float
    max_seq_length: int
    reasoning: str


class EpochLog(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    elapsed_sec: float


class TrainingState(TypedDict):
    logs: List[EpochLog]
    checkpoint_path: str
    adapter_path: str
    total_epochs_run: int
    best_loss: float
    best_epoch: int
    stop_reason: Optional[str]
    training_time_sec: float


class EvalResults(TypedDict):
    bleu: float
    rouge_l: float
    perplexity: float
    exact_match: float
    primary_metric: str
    primary_value: float
    eval_time_sec: float
    samples_eval: int


class DecisionRecord(TypedDict):
    iteration: int
    decision: str
    reasoning: str
    metric_at_decision: float
    suggested_changes: Optional[Dict[str, Any]]


class ReportState(TypedDict):
    json_path: str
    md_path: str
    adapter_path: str


class OrchestrationState(TypedDict):
    run_config: RunConfig
    dataset: DatasetState
    current_hyperparams: HyperparamConfig
    hyperparam_reasoning: str
    training: TrainingState
    eval_results: EvalResults
    eval_history: Annotated[List[EvalResults], operator.add]
    hyperparam_history: Annotated[List[HyperparamConfig], operator.add]
    decision_history: Annotated[List[DecisionRecord], operator.add]
    training_logs: Annotated[List[EpochLog], operator.add]
    agent_logs: Annotated[List[str], operator.add]
    decision: str
    decision_reasoning: str
    suggested_hyperparam_changes: Optional[Dict[str, Any]]
    iteration: int
    report: ReportState
    status: str
    current_agent: str
    error: Optional[str]