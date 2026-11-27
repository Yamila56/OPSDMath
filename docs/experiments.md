# Experiments

The main comparison uses the frozen Qwen3-8B reference, two OPSD objectives and the earlier equation-chain SFT controls. The 192 confirmation questions come from the GSM8K training partition and were held out from project training and selection. All methods generate under the same combined prompt. The 128-token and 512-token scores come from one trajectory. Early retention uses questions the frozen model answers by 128 tokens. Late acceleration uses questions the base answers only at 512 tokens.

## Qwen3-8B results by seed

| Method | 128-token correct / 192 | Early retained / 140 | Late accelerated / 41 | 512-token correct / 192 |
| --- | ---: | ---: | ---: | ---: |
| Frozen Qwen3-8B | 140 | 140 | 0 | 181 |
| Clipped OPSD | 94 / 105 | 94 / 103 | 0 / 2 | 181 / 181 |
| **Full forward KL** | **157 / 156** | **138 / 138** | 18 / 17 | **182 / 182** |
| Equation-chain SFT | 140 / 137 | 112 / 111 | **25 / 23** | 143 / 140 |

Slashes separate seeds 17 and 29. The frozen model uses one shared prediction trajectory.

The paired analysis resamples questions and reports seed-specific effects and the seed-averaged effect. At 128 tokens, full KL exceeded the frozen base by 8.59 pp and equation SFT by 9.38 pp. The paired 512-token comparison with the base includes zero. On SVAMP, all three of the base and two full-KL runs scored 236/256 at 128 tokens. The full-KL runs retained 233 and 234 of the 236 original early answers. The SVAMP base has two late questions under this protocol.

## Cross-scale EOS audit

The local logit audit found that clipping would lower EOS log-probability at 32/59 reference endings for frozen 8B, compared with 2/59 for frozen 1.7B. The two 8B adapters gave 26/59 and 22/59. Full KL gave zero such endings in these groups. The audit measures a local derivative. The share of the generation gap attributable to EOS remains open. [Gradient and prediction archives](../results/gradient_audit) · [Paired results and checksums](../results/kl_ablation)

| Checkpoint | Qwen3-1.7B EOS decreases | Qwen3-8B EOS decreases |
| --- | ---: | ---: |
| Frozen base | 2/59 | 32/59 |
| Adapter, seed 17 | 2/59 | 26/59 |
| Adapter, seed 29 | 1/59 | 22/59 |

The corresponding full-KL counts are zero in all six runs. [Archived direction counts](../results/gradient_audit/eos-paired.json)

## Formal studies

| Study | Code | Evidence |
| --- | --- | --- |
| Matched objective ablation and equation SFT control | [kl_ablation](../experiments/kl_ablation) | [Paired statistics and SHA-256-verified prediction bundles](../results/kl_ablation) |
| Cross-scale EOS direction and logit gradients | [gradient_audit](../experiments/gradient_audit) | [Per-position records](../results/gradient_audit) |
| SVAMP transfer under the same token budgets | [transfer_evaluation](../experiments/transfer_evaluation) | [Accuracy and retention](../results/transfer_evaluation/summary.json) · [GPU recipe](../configs/evaluation_svamp_8b.json) |
| Upstream parity and interrupted-run recovery | [trainer_validation](../experiments/trainer_validation) | [Audit results](../results/trainer_validation) |

The [supplementary studies](supplementary.md) document related experiments and link to their source and evidence.
