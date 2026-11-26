# OPSDMath

OPSDMath studies how on-policy self-distillation affects short-budget math answers. A LoRA student samples its own trajectories, and the same frozen Qwen3 model scores those prefixes with access to the reference answer.

## Method

![Online student and frozen teacher scoring shared prefixes](assets/figures/overview.svg)

The matched Qwen3-8B experiment changes only the vocabulary-wise KL cap. Both arms use 256 training questions, 64 updates and a 128-token rollout cap. [Method](docs/method.md)

## Main results

| Method | 128-token correct / 192 | Early retained / 140 | Late accelerated / 41 | 512-token correct / 192 |
| --- | ---: | ---: | ---: | ---: |
| Frozen Qwen3-8B | 140 | 140 | 0 | 181 |
| Clipped OPSD | 94 / 105 | 94 / 103 | 0 / 2 | 181 / 181 |
| **Full forward KL** | **157 / 156** | **138 / 138** | 18 / 17 | **182 / 182** |
| Equation-chain SFT | 140 / 137 | 112 / 111 | **25 / 23** | 143 / 140 |

The two values are seeds 17 and 29 on the same project-held-out GSM8K questions. Full KL improved 128-token accuracy against clipping by **29.69 pp** with a question-paired 95% interval of **[23.70, 35.68] pp**. [Experiment protocol and evidence](docs/experiments.md)

![Short and long budget accuracy, behavioral trade-off and paired effects](assets/figures/main_results.svg)

## Quick start

Use Python 3.12 for the CPU replay.

```bash
python -m pip install -e .
opsdmath-reproduce --experiment kl-ablation
```

GPU training and inference use the pinned training dependencies and the recipes in `configs/`. [Reproduction commands](docs/reproduction.md)

## References

[Self-Distilled Reasoner](https://proceedings.mlr.press/v306/zhao26be.html) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [GSM8K](https://github.com/openai/grade-school-math) · [SVAMP](https://aclanthology.org/2021.naacl-main.168/)