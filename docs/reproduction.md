# Reproduction

## CPU evidence replay

Use Python 3.12 or later. From the checkout root, run `python -m pip install -e .` and `opsdmath-reproduce --experiment kl-ablation`. The replay unpacks the saved predictions to a temporary directory, verifies the original archive hashes, and compares the recomputed paired report with the published JSON. It also recounts the EOS audit and verifies SVAMP early retention directly from paired per-question scores. Use `--output` to keep a new report. The archive also contains the evaluation inputs, labels and per-seed training configurations. [Evidence directory](../results/kl_ablation)

Run `opsdmath-reproduce --experiment svamp-transfer` for the standalone transfer report. The [shell wrapper](../scripts/reproduce.sh) accepts the same options.

## Local environment

The package supports Python 3.12 or later. The setup script supports Linux and Git Bash on Windows. It uses the selected Python interpreter to install the checkout into a virtual environment. Run `cpu` mode for the CPU replay and test dependencies. Run `cuda` mode for CUDA 12.6 and the pinned training dependencies. Choose a separate virtual environment for each if you use both.

```bash
PYTHON=python3.12 bash scripts/setup_local.sh cpu .venv-cpu
PYTHON=python3.12 bash scripts/setup_local.sh cuda .venv
source .venv/bin/activate
```

On Windows, select a Python 3.12+ interpreter with PYTHON=python and activate with `source .venv/Scripts/activate`. The editable install method remains available with a preconfigured PyTorch environment.

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

The evaluator reuses a group only when its protocol and the hashes of its tokens, both score files, and final metrics match. A missing or damaged output triggers regeneration. An older run without a checksum manifest is regenerated once during evaluation.

## SVAMP transfer inference

Prepare the same 256 questions used in the transfer study. The frozen protocol skips the first 256 sorted IDs after normalized GSM8K-training exclusion and deduplication. The recipe pins the source revision, both input hashes, and the original batch size of 64.

```bash
python -m opsdmath.data.svamp data/train.jsonl data/svamp-transfer --skip-first 256
opsdmath-evaluate --config configs/evaluation_svamp_8b.json \
  --model models/qwen3-8b --data data/svamp-transfer --output runs/svamp-transfer \
  --group base full-17=runs/full-17/adapter full-29=runs/full-29/adapter
```

Train both full-KL adapters with the recipe above using seeds 17 and 29, or supply the respective original weights. Each group produces one greedy 512-token trajectory per question and scores its 128-token prefix. [Transfer results](../results/transfer_evaluation/summary.json)

## Cross-scale gradient audit

The audit uses 64 GSM8K development questions from the pinned source. The archived split-preparation command selects sorted IDs 384 through 447 after excluding test overlap. The audit also uses the 256-question training split for reference-coverage statistics.

```bash
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir models/qwen3-1.7b
python -m archive.experiments.budget_efficiency.opsd_calibration prepare \
  --train data/train.jsonl --output data/gradient-audit
python -m experiments.gradient_audit.audit_privileged_gradients \
  --small models/qwen3-1.7b --large models/qwen3-8b \
  --dev data/gradient-audit/reference.jsonl --train data/train.jsonl \
  --output runs/eos-frozen
```

To audit the adapted models, use the same command with a fresh output directory and one `--adapter SCALE SEED PATH` argument for each adapter. These arguments select adapter audits in place of the frozen checkpoints. For example, the four published checkpoint conditions use the arguments below. The paths should contain the adapters corresponding to the archived gradient audit.

```bash
python -m experiments.gradient_audit.audit_privileged_gradients \
  --small models/qwen3-1.7b --large models/qwen3-8b \
  --dev data/gradient-audit/reference.jsonl --train data/train.jsonl \
  --adapter 1.7B 17 adapters/audit-1p7b-17 --adapter 1.7B 29 adapters/audit-1p7b-29 \
  --adapter 8B 17 adapters/audit-8b-17 --adapter 8B 29 adapters/audit-8b-29 \
  --output runs/eos-adapted
```

## Small-model GPU smoke test

The smoke script checks prompt parity, performs one Qwen3-1.7B optimizer update on four examples, then reloads the adapter for prediction. It requires the pinned upstream collator checkout and explicit local model, data, and output paths. [Upstream revision and validation](trainer_parity.md)

```bash
git clone https://github.com/siyan-zhao/OPSD upstream
git -C upstream checkout 0622e0ae62eadfa869ad1397a960b40044ab7388
PYTHON=.venv/bin/python bash scripts/preflight.sh models/qwen3-1.7b data runs/preflight
```

## Local checkpoint recovery

Resume online OPSD or SFT with `python -m opsdmath.training.trainer` and the original training arguments, using a fresh output directory and `--resume-from` pointing to `train-state.pt`. The checkpoint must match the original model path, ordered data, optimization settings and rollout mode. For an older state file, keep its original `config.json` alongside it so the runner can verify the device and online rollout mode. Fixed-rollout experiments may save an adapter for evaluation.

## Figures

Install `.[figures]` and run `python scripts/render_figures.py` to render the three README figures. Use `--output` to render to a separate directory.

## Licenses

The published question data comes from GSM8K and SVAMP. [GSM8K license](../licenses/GSM8K.txt) · [SVAMP license](../licenses/SVAMP.txt)

The upstream Trainer is a separate checkout. The pinned revision has no repository-wide license file. The dataset notices apply to their respective datasets. A license covering the project requires a separate review of the rights to each component. [Pinned OPSD source](https://github.com/siyan-zhao/OPSD/tree/0622e0ae62eadfa869ad1397a960b40044ab7388)
