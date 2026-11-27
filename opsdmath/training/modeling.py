"""Load the shared Qwen3 backbone and configure its LoRA adapters."""


def lora_alpha(rank):
    return 2 * rank


def lora_config(rank):
    from peft import LoraConfig

    return LoraConfig(
        r=rank, lora_alpha=lora_alpha(rank), task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def load_model(path, tokenizer, device):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": str(device)}, attn_implementation="sdpa", local_files_only=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def load_eval_model(path):
    import torch

    tokenizer = load_tokenizer(path)
    model = load_model(path, tokenizer, torch.device("cuda:0"))
    return model.eval(), tokenizer
