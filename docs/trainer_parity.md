# OPSD implementation and Trainer checks

The single-device implementation uses the objective published by the [Self-Distilled Reasoner authors](https://github.com/siyan-zhao/OPSD) at commit `0622e0ae62eadfa869ad1397a960b40044ab7388`. Check out that revision separately and supply its directory via `--upstream` to the parity and recovery runners. The loss sums forward KL terms after clipping each vocabulary contribution at 0.05. Clipping changes the sign of some summed losses. The training probe logs unclipped KL and the fraction of terms and positions affected by clipping.

## Fixed-update comparison

Both branches start from the same Qwen3-1.7B and LoRA weights. They use four identical, unpadded cached trajectories, frozen teachers, an AdamW step, and the same gradient clipping.

| Upstream `compute_loss` configuration | Maximum absolute loss difference | Maximum gradient difference | Maximum updated weight difference |
| --- | ---: | ---: | ---: |
| Default BF16 scoring | 9.03×10⁻⁴ | 2.98×10⁻³ | 1.00×10⁻⁵ |
| FP32 objective, upstream forward inputs | 6.50×10⁻⁴ | 2.92×10⁻³ | 1.00×10⁻⁵ |
| FP32 objective, no redundant mask for these unpadded examples | 9.32×10⁻¹⁰ | 4.29×10⁻⁵ | 9.97×10⁻⁶ |
| FP32 objective, matching unpadded inputs and completion-only logits | 9.32×10⁻¹⁰ | 0 | 0 |

The comparison changes the input mask and output-logit scope of the actual upstream `compute_loss` call one at a time. Matching all three settings gives identical updated parameters for all 392 LoRA tensors in this audit. Padded batches still require an attention mask. The upstream Trainer keeps its own data collator and prompt format in the independent online test.

## Online training and recovery

On TRL 0.26.0 with Transformers generation, the deterministic two-update test produces identical prompt and rollout token hashes and final LoRA parameters after interruption and recovery. The default settings can produce small trajectory and weight differences.

An expanded run uses 32 GSM8K examples, four microsteps per update, eight optimizer updates, and a 16-token rollout cap. After restarting at update four, the deterministic run matches all 32 prompt and rollout hashes and all 392 LoRA tensors at updates four and eight. This verifies accumulated recovery for the tested upstream Trainer and Transformers path.

Run the checks from the checkout root with the pinned upstream environment. The fixed-update test consumes its starting adapter and matching token-ID trajectories. For the recovery test, run once to save the midpoint checkpoint, again with `--resume` in the same directory, then with `--uninterrupted` and a separate output directory.

```bash
python -m experiments.trainer_validation.check_trainer_update --help
python -m experiments.trainer_validation.smoke_upstream_trainer --help
```

[Fixed-update implementation](../experiments/trainer_validation/check_trainer_update.py) · [Online recovery script](../experiments/trainer_validation/smoke_upstream_trainer.py) · [Numerical results](../results/trainer_validation/engineering-followup.json)
