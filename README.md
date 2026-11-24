# Short-Answer Math Distillation

How does on-policy self-distillation change the answers a small language model can finish within a fixed output budget? This study trains Qwen3-1.7B on 256 GSM8K problems and compares two post-training methods on a shared set of 256 held-out questions.

The student generates its own solution. A frozen copy of the base model sees the reference solution and supplies token probabilities along the student's trajectory. The SFT comparison learns directly from the same references. Both methods use LoRA and make 64 optimizer updates on a single GPU.

## Results

| Model | Correct | Parseable answers | Mean output tokens |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B | 144/256 (56.25%) | 181/256 | 151.2 |
| Supervised fine-tuning | 154/256 (60.16%) | 192/256 | 168.7 |
| On-policy self-distillation | **175/256 (68.36%)** | **235/256** | **134.6** |

With 256 new tokens per question, OPSD improved exact numeric accuracy over SFT by 8.20 percentage points under the primary scoring rule. Its outputs reached the token cap on 19 questions, compared with 41 for SFT. A retrospective check that also accepts inline final answers and common unit suffixes reduced the difference to 3.91 points, with a paired 95% interval of −1.95 to +9.77 points. The table reports the primary training seed. [Metrics](results/results.json) · [Experiment report and case analysis](实验报告.md)

A second training seed produced a 3.91-point difference under the primary rule, with a paired 95% interval of −2.34 to +10.16 points. Both methods scored 169/256 under the broader format rule in that run. OPSD produced shorter outputs under both seeds. [Second-seed metrics](results/seed29/results.json)

## Reproduce

Use Python 3.11 or 3.12 with CUDA-enabled PyTorch 2.8. The training environment uses Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.11.0 and safetensors 0.5.3.

```bash
python -m pip install transformers==4.57.1 peft==0.17.1 accelerate==1.11.0 safetensors==0.5.3
python gsm8k.py prepare data
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir model

python run_math.py opsd --model model --data data/train.jsonl --output runs/train-opsd
python run_math.py sft --model model --data data/train.jsonl --output runs/train-sft

python run_math.py predict --model model --data data/test_inputs.jsonl --output runs/eval-base --batch-size 16
for group in sft opsd; do
    python run_math.py predict --model model --data data/test_inputs.jsonl \
        --adapter "runs/train-$group/adapter" --prediction-name "$group" \
        --output "runs/eval-$group" --batch-size 16
done
python gsm8k.py score data/test_labels.jsonl \
    runs/eval-base/base.jsonl runs/eval-sft/sft.jsonl runs/eval-opsd/opsd.jsonl
```

Training and evaluation for the primary run took approximately 22 minutes on one A800. The repository includes the dataset hashes and predictions from both training seeds. [Dataset manifest](results/data-manifest.json) · [Seed 17 predictions](results/seed17) · [Seed 29 predictions](results/seed29)

## References

The distillation objective follows *Self-Distilled Reasoner* by Zhao et al. The implementation here adapts it to an independently constructed GSM8K split, short generation and a one-device training loop. [Paper](https://proceedings.mlr.press/v306/zhao26be.html) · [Reference implementation](https://github.com/thinkingmanyangyang/OPSD/tree/0622e0ae62eadfa869ad1397a960b40044ab7388)

The base model and benchmark are provided by their respective authors. [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [GSM8K](https://github.com/openai/grade-school-math)
