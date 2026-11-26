# Reproduction

## CPU evidence replay

From the checkout root, run `python -m pip install -e .` and `opsdmath-reproduce --experiment kl-ablation`. This unpacks the saved predictions to a temporary directory, recomputes the pairings and checks the complete JSON against the published file. Use `--output` to keep a new report. The archive also contains the evaluation inputs, labels and per-seed training configurations. [Evidence directory](../results/scale-transfer/privileged-gradient-audit/kl-pilot)

## Reference training and inference

Use a CUDA-enabled PyTorch 2.8 install with `python -m pip install -e '.[training]'`. The recorded Qwen3-8B model revision is `b968826d9c46dd6066d109eabc6255188de91218`. The configs record the model index hash, seed streams, training arguments and paired evaluation protocol.

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/qwen3-8b
python gsm8k.py prepare data
opsdmath-train --config configs/opsd_8b_full_kl.json --seed 17 --model models/qwen3-8b --data data/train.jsonl --output runs/full-17
```

For clipped OPSD use `configs/opsd_8b_clipped_kl.json`. For the equation control, run `python experiments/legacy/short_sft_baseline.py prepare data/train.jsonl data/equations --target equations` and pass `data/equations/train.jsonl` with `configs/equation_sft_8b.json`. The published controls use seeds 17 and 29. Historical trainer flags are also available via `python run_math.py --help`.

Extract `data/inputs.jsonl` and `data/labels.jsonl` from the seed 29 archive into an evaluation directory. Point `--data` to the directory containing those two files. The evaluation checks both hashes and uses the same greedy 512-token generation and 128-token prefix score.

```bash
opsdmath-evaluate --config configs/evaluation_8b.json --model models/qwen3-8b --data path/to/extracted/data --output runs/kl-eval --group base full17=runs/full-17/adapter
```

The original full-KL adapters are kept separately. Their SHA-256 records are in the evidence directory. The archived shell and Python studies assume an editable installation and the working directory indicated in the source launcher. [Trainer parity](trainer_parity.md)

## Figures

Install `.[figures]` and run `python scripts/figures/render.py` to regenerate the archived figures.
The published question data comes from GSM8K and SVAMP. [GSM8K license](../licenses/GSM8K.txt) · [SVAMP license](../licenses/SVAMP.txt)
