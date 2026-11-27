# Reproduction

## CPU evidence replay

Use Python 3.12 or later. From the checkout root, run `python -m pip install -e .` and `opsdmath-reproduce --experiment kl-ablation`. The replay unpacks the saved predictions to a temporary directory, verifies the original archive hashes, and compares the recomputed paired report with the published JSON. It also recounts the EOS audit and verifies SVAMP early retention directly from paired per-question scores. Use `--output` to keep a new report. The archive also contains the evaluation inputs, labels and per-seed training configurations. [Evidence directory](../results/kl_ablation)

Run `opsdmath-reproduce --experiment svamp-transfer` for the standalone transfer report. The [shell wrapper](../scripts/reproduce.sh) accepts the same options.

## Reference training and inference

Use a CUDA-enabled PyTorch 2.8 install with `python -m pip install -e '.[training]'`. The recorded Qwen3-8B model revision is `b968826d9c46dd6066d109eabc6255188de91218`. The configs record the model index hash, seed streams, training arguments and paired evaluation protocol. For training, the active flags come from the recipe arguments and selected seed. The runner checks the recorded update count, optimizer, schedule, LoRA alpha and OPSD sampling settings against the effective protocol.

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/qwen3-8b
python scripts/prepare_data.py data
opsdmath-train --config configs/full_kl_8b.json --seed 17 --model models/qwen3-8b --data data/train.jsonl --output runs/full-17
```

For clipped OPSD use `configs/clipped_kl_8b.json`. For the equation control, run `python -m experiments.kl_ablation.short_sft_baseline prepare data/train.jsonl data/equations --target equations` and pass `data/equations/train.jsonl` with `configs/equation_sft_8b.json`. The published controls use seeds 17 and 29. Historical trainer flags are also available via `python -m opsdmath.training.trainer --help`.

Extract `data/inputs.jsonl` and `data/labels.jsonl` from the seed 29 archive into an evaluation directory. Point `--data` to the directory containing those two files. The evaluator verifies both hashes and the expected question count. It validates the fixed combined prompt, disabled thinking, and 128/512-token budget fields in the evaluation recipe. A greedy 512-token trajectory supplies both scores, with a direct short-decode check on the first batch.

```bash
opsdmath-evaluate --config configs/evaluation_8b.json --model models/qwen3-8b --data path/to/extracted/data --output runs/kl-eval --group base full17=runs/full-17/adapter
```

The repository contains scored predictions, training recipes, and adapter hashes. To reproduce model outputs, train each adapter locally or use original weights obtained separately. An adapter directory needs both `adapter_config.json` and `adapter_model.safetensors`. The original uncompressed checkpoints and compressed full-KL checkpoints have separate SHA-256 records. [Uncompressed checkpoints](../results/kl_ablation/cloud-adapter-sha256.txt) · [Compressed full-KL checkpoints](../results/kl_ablation/full-kl-persistent-sha256.txt) The study scripts assume an editable installation. Run Python study modules from the checkout root. See the source of each shell launcher for its input paths and environment. [Trainer parity](trainer_parity.md)

### Local checkpoint recovery

Resume online OPSD or SFT with `python -m opsdmath.training.trainer` and the original training arguments, using a fresh output directory and `--resume-from` pointing to `train-state.pt`. The checkpoint must match the original model path, ordered data, optimization settings and rollout mode. For an older state file, keep its original `config.json` alongside it so the runner can verify the device and online rollout mode. Fixed-rollout experiments may save an adapter for evaluation.

## Figures

Install `.[figures]` and run `python scripts/render_figures.py` to render the three README figures. Use `--output` to render to a separate directory.

## Licenses

The published question data comes from GSM8K and SVAMP. [GSM8K license](../licenses/GSM8K.txt) · [SVAMP license](../licenses/SVAMP.txt)

The upstream Trainer is a separate checkout. The pinned revision has no repository-wide license file. The dataset notices apply to their respective datasets. A license covering the project requires a separate review of the rights to each component. [Pinned OPSD source](https://github.com/siyan-zhao/OPSD/tree/0622e0ae62eadfa869ad1397a960b40044ab7388)
