import torch
from typing import Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

TARGET_MODULES = {
    "qwen2.5": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def get_model_family(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if "qwen" in model_id_lower:
        return "qwen2.5"
    elif "llama" in model_id_lower:
        return "llama"
    elif "mistral" in model_id_lower:
        return "mistral"
    elif "phi" in model_id_lower:
        return "phi"
    elif "gemma" in model_id_lower:
        return "gemma"
    return "llama"


def load_model(model_id: str, quantization: bool = True) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if quantization else None,
    )
    return model, tokenizer


def apply_lora(model, hyperparams: dict, model_family: str) -> object:
    target_modules = TARGET_MODULES.get(model_family, TARGET_MODULES["llama"])
    lora_config = LoraConfig(
        r=hyperparams["lora_r"],
        lora_alpha=hyperparams["lora_alpha"],
        lora_dropout=hyperparams["lora_dropout"],
        target_modules=target_modules,
        bias=hyperparams.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def save_adapter(model, path: str) -> str:
    import os
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    return path


def load_adapter(base_model_id: str, adapter_path: str, quantization: bool = True):
    model, tokenizer = load_model(base_model_id, quantization)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer
