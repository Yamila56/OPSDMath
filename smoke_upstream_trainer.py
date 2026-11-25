"""Run and resume a single-device smoke test of pinned upstream OPSD."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, enable_full_determinism, set_seed
from trl.experimental.gold.gold_config import GOLDConfig

from gsm8k import read_rows


class StopAtUpdate(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True
            control.should_save = True
        return control


def compare_adapters(left, right):
    one, two = (load_file(path/"adapter_model.safetensors") for path in (left, right))
    if one.keys() != two.keys():
        raise RuntimeError("Compared adapters have different tensors")
    differences = [float((one[k]-two[k]).abs().max()) for k in one]
    return dict(tensors=len(one), changed=sum(delta > 0 for delta in differences),
                max_abs_difference=max(differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "upstream", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--uninterrupted", type=Path, metavar="RESUMED_RUN",
                      help="Run without a restart and compare its final adapter to a resumed run.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--rollout-cap", type=int, default=48)
    args = parser.parse_args()
    if args.steps < 2 or args.steps % 2 or not 1 <= args.accumulation or not 1 <= args.rollout_cap < 384:
        parser.error("Use an even number of updates, positive accumulation, and rollout cap from 1 to 383")
    midpoint = args.steps//2
    examples = read_rows(args.data)[:max(4, args.steps*args.accumulation)]
    if len(examples) < args.steps*args.accumulation:
        parser.error("Provide at least one distinct row per microstep")
    sys.path.insert(0, str(args.upstream.resolve()))
    from opsd_trainer import OPSDTrainer

    def digest(tokens):
        return hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()

    class RolloutAuditTrainer(OPSDTrainer):
        def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
            generated = super().generate_on_policy_outputs(model, inputs, generation_config, pad_token_id)
            ids = generated[0]
            prompt_length = inputs["student_prompt_length"]
            traces.append(dict(step=self.state.global_step+1, prompt_sha256=digest(ids[:, :prompt_length]),
                               rollout_sha256=digest(ids[:, prompt_length:])))
            return generated

    (enable_full_determinism if args.deterministic else set_seed)(17)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    cfg = GOLDConfig(
        output_dir=str(args.output), beta=0, lmbda=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accumulation, max_steps=args.steps, max_length=512,
        max_completion_length=args.rollout_cap,
        use_vllm=False, learning_rate=5e-6, bf16=True, temperature=1.1, top_p=.95, top_k=20,
        max_grad_norm=.1, logging_steps=1, save_steps=midpoint, report_to="none",
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True}, seed=17,
        full_determinism=args.deterministic,
    )
    peft = LoraConfig(r=64, lora_alpha=128, task_type="CAUSAL_LM", target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    interrupted = not (args.resume or args.uninterrupted)
    traces = []
    trainer = RolloutAuditTrainer(model=model, args=cfg, processing_class=tokenizer, peft_config=peft,
                                 train_dataset=Dataset.from_list(examples),
                                 fixed_teacher=True, jsd_token_clip=.05,
                                 callbacks=[StopAtUpdate(midpoint)] if interrupted else [])
    start = str(args.output/f"checkpoint-{midpoint}") if args.resume else None
    trainer.train(resume_from_checkpoint=start)
    expected = midpoint if interrupted else args.steps
    if trainer.state.global_step != expected:
        raise RuntimeError(f"Expected optimizer step {expected}, got {trainer.state.global_step}")
    checkpoint = args.output/f"checkpoint-{expected}"
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"):
        if not (checkpoint/name).is_file():
            raise RuntimeError(f"Missing checkpoint component {name}")
    log = trainer.state.log_history
    if not any("loss" in event and event["step"] == expected for event in log):
        raise RuntimeError("Expected a logged distillation loss after the update")
    result = dict(global_step=expected, resumed=args.resume, deterministic=args.deterministic,
                  gradient_accumulation_steps=args.accumulation, total_training_rows=len(examples),
                  max_completion_length=args.rollout_cap,
                  peak_mib=torch.cuda.max_memory_allocated()/2**20, rollouts=traces, log=log)
    if expected == args.steps:
        result["optimizer_update"] = compare_adapters(args.output/f"checkpoint-{midpoint}", checkpoint)
        if not result["optimizer_update"]["changed"]:
            raise RuntimeError("The second optimizer step did not change any adapter weights")
    if args.uninterrupted:
        result["midpoint_versus_resumed"] = compare_adapters(args.uninterrupted/f"checkpoint-{midpoint}",
                                                            args.output/f"checkpoint-{midpoint}")
        result["versus_resumed"] = compare_adapters(args.uninterrupted/f"checkpoint-{args.steps}", checkpoint)
    (args.output/f"step-{expected}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
