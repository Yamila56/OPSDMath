# Reproduction

## CPU evidence replay

From the checkout root, run `python -m pip install -e .` and `opsdmath-reproduce --experiment kl-ablation`. This unpacks the saved predictions to a temporary directory, recomputes the pairings and checks the complete JSON against the published file. Use `--output` to keep a new report. The archive also contains the evaluation inputs, labels and per-seed training configurations. [Evidence directory](../results/scale-transfer/privileged-gradient-audit/kl-pilot)

## Reference training and inference

Use a CUDA-enabled PyTorch 2.8 install with `python -m pip install -e '.[training]'`. The recorded Qwen3-8B model revision is `b968826d9c46dd6066d109eabc6255188de91218`. The configs record the model index hash, seed streams, training arguments and paired evaluation protocol. For training, the active flags come from the recipe arguments and selected seed. The runner checks the recorded update count, optimizer, schedule, LoRA alpha and OPSD sampling settings against the effective protocol.

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/qwen3-8b
python gsm8k.py prepare data
opsdmath-train --config configs/opsd_8b_full_kl.json --seed 17 --model models/qwen3-8b --data data/train.jsonl --output runs/full-17
```

For clipped OPSD use `configs/opsd_8b_clipped_kl.json`. For the equation control, run `python -m experiments.sft_baselines.short_sft_baseline prepare data/train.jsonl data/equations --target equations` and pass `data/equations/train.jsonl` with `configs/equation_sft_8b.json`. The published controls use seeds 17 and 29. Historical trainer flags are also available via `python run_math.py --help`.

Extract `data/inputs.jsonl` and `data/labels.jsonl` from the seed 29 archive into an evaluation directory. Point `--data` to the directory containing those two files. The evaluation checks both hashes and uses the same greedy 512-token generation and 128-token prefix score.

```bash
opsdmath-evaluate --config configs/evaluation_8b.json --model models/qwen3-8b --data path/to/extracted/data --output runs/kl-eval --group base full17=runs/full-17/adapter
```

The original full-KL adapters are kept separately. Their SHA-256 records are in the evidence directory. The study scripts assume an editable installation. Run Python study modules from the checkout root. See the source of each shell launcher for its input paths and environment. [Trainer parity](trainer_parity.md)

### Local checkpoint recovery

Resume online OPSD or SFT with [run_math.py](../run_math.py) and the original training arguments, using a fresh output directory and `--resume-from` pointing to `train-state.pt`. The checkpoint must match the original model path, ordered data, optimization settings and rollout mode. For an older state file, keep its original `config.json` alongside it so the runner can verify the device and online rollout mode. Fixed-rollout experiments may save an adapter for evaluation.

## Figures

Install `.[figures]` and run `python scripts/figures/render.py` to regenerate the archived figures.
The published question data comes from GSM8K and SVAMP. [GSM8K license](../licenses/GSM8K.txt) · [SVAMP license](../licenses/SVAMP.txt)

The upstream Trainer is a separate checkout. The pinned revision has no repository-wide license file. The dataset notices apply to their respective datasets. A license covering the project requires a separate review of the rights to each component. [Pinned OPSD source](https://github.com/siyan-zhao/OPSD/tree/0622e0ae62eadfa869ad1397a960b40044ab7388)
