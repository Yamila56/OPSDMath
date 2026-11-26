# OPSDMath

**Gradient Analysis and Budget-Efficient On-Policy Self-Distillation**

OPSDMath studies how post-training changes which math problems a Qwen model can answer within a fixed token budget. It combines a single-GPU implementation of on-policy self-distillation (OPSD), a token-level gradient audit, and a matched training experiment on Qwen3-8B.

[Main results](#qwen3-8b-kl-ablation) · [Trainer checks](#training-and-gradient-audit) · [Reproduce](#recompute-the-published-results) · [Next experiments](NEXT_EXPERIMENTS.md)

## Qwen3-8B KL ablation

**Removing vocabulary-wise KL clipping improved 128-token numeric accuracy by 29.69 percentage points** on the 192-question project holdout. The question-paired 95% interval was **[23.70, 35.68] pp**, averaged over seeds 17 and 29. Each matched pair used the same 256 training questions, question order, rollout seed, 64 optimizer updates, answer-only teacher, and 128-token training cap. Online trajectories could diverge as the student weights changed.

| Method | 128-token correct / 192 | Early answers retained / 140 | Late answers accelerated / 41 | 512-token correct / 192 |
| --- | ---: | ---: | ---: | ---: |
| Frozen Qwen3-8B | 140 | 140 | 0 | 181 |
| OPSD with 0.05 vocabulary-wise clipping | 94 / 105 | 94 / 103 | 0 / 2 | 181 / 181 |
| **OPSD with full forward KL** | **157 / 156** | **138 / 138** | 18 / 17 | **182 / 182** |
| Equation-chain SFT | 140 / 137 | 112 / 111 | **25 / 23** | 143 / 140 |

Slashes separate seeds 17 and 29. All models use the same evaluation prompt. The 128- and 512-token scores come from the same generated trajectory. Early questions are correct at 128 tokens for the frozen base, and late questions are only correct at 512 tokens for that base. “Accelerated” means the student delivered a correct answer by the short cutoff. The proposed follow-up checks reasoning validity separately from answer timing.

Full KL averaged **+8.59 pp [4.43, 13.28]** over the frozen base and **+9.38 pp [2.86, 15.89]** over the existing equation-chain SFT controls at 128 tokens. SFT accelerated more late questions, while full KL retained more early and long-budget successes. At 512 tokens the paired interval against the base includes zero.

On **SVAMP**, the two full-KL runs and the base each reached **236/256** at 128 tokens. The runs retained **233/236** and **234/236** of the original early answers and scored **237/256** and **238/256** at 512 tokens, compared with **238/256** for the base. Only two questions were late for the base under this protocol, so cross-source acceleration remains open.

The 192 questions come from the original GSM8K training partition and were held out from project training and earlier model selection. The two seeds share those same questions. The controlled result identifies a cost of clipping in this 8B recipe. Its applicability to other model families and training schedules is open. [Experimental report](8B规模对照报告.md#第二种子算式-sft-与题源迁移) · [Paired scores](results/scale-transfer/privileged-gradient-audit/kl-pilot/paired-two-seed.json) · [Split manifest](results/scale-transfer/privileged-gradient-audit/kl-pilot/data-manifest.json)

## Training and gradient audit

- **Online rollouts and a shared frozen teacher.** Before each optimizer update, the current LoRA student generates a batch of trajectories. With its adapter disabled, the same base model scores the identical completion positions under a prompt that includes the known answer. Evaluation prompts withhold the reference answer. Full forward KL uses the entire vocabulary at each scored position. [Single-GPU implementation](run_math.py)
- **Local EOS update direction.** At the end of 59 verified compact references, a local descent step on logits under vocabulary-wise clipping would decrease the EOS log-probability at **32/59** positions for frozen 8B and **2/59** for frozen 1.7B. The two existing 8B checkpoints gave **26/59** and **22/59**. Full KL raised it at all these reference endings. The analytic derivatives were checked with autograd. How much of the generation gap this mechanism accounts for remains open. [Audit code](audit_privileged_gradients.py) · [Position-level results](results/scale-transfer/privileged-gradient-audit)
- **Numerical parity and checkpoint recovery.** On four cached, unpadded 1.7B trajectories with matched FP32 objective, attention inputs, and completion-logit scope, the maximum loss difference from the pinned upstream Trainer was **9.32 × 10⁻¹⁰**, with identical gradients and updates across **392 LoRA tensors**. In a separate deterministic upstream test with 32 examples and eight optimizer updates, recovery from update four reproduced all rollout hashes and adapter tensors. [Trainer parity and recovery](TRAINER_PARITY.md)

## Recompute the published results

The archives include predictions, scores, and the matched OPSD run configurations. The following Bash commands replay the published question-level statistics on CPU with Python and NumPy, from the repository root.

```bash
pilot=results/scale-transfer/privileged-gradient-audit/kl-pilot
replay=$(mktemp -d)
mkdir -p "$replay/seed17"

tar -xzf "$pilot/seed29-evidence.tar.gz" -C "$replay"
tar -xzf "$pilot/seed17-evidence.tar.gz" -C "$replay/seed17"
tar -xzf "$pilot/controls-evidence.tar.gz" -C "$replay/seed17"
PYTHONPATH=. python "$pilot/recompute-two-seed.py" "$replay"
diff -u "$pilot/paired-two-seed.json" "$replay/paired-two-seed.json"
```

For a model rerun, use the archived `clipped/config.json` and `full/config.json`, the evaluation `protocol.json` files, and the [8B environment and training instructions](8B规模对照报告.md#复现). In the matched OPSD comparison, `--kl-clip 0.05` selects vocabulary-wise clipping and `--kl-clip none` selects full forward KL. The repository supplies the results archives and adapter checksums. Re-running inference with the original full-KL weights requires the separately stored LoRA adapters. [Evidence archives and checksums](results/scale-transfer/privileged-gradient-audit/kl-pilot)

## Earlier studies

| Study | What it established |
| --- | --- |
| [1.7B budgets and shared answer field](后续实验报告.md) | The 128-token numeric advantage of answer-only OPSD over equation-chain SFT persisted when both used the same grammar-constrained answer field. [Original reproduction commands](https://github.com/Yamila56/OPSDMath/blob/dfab64e036c221c5d04e683fab4b1e68a0a50b61/README.md#reproduce-the-comparison) |
| [Condition changes and teacher notes](条件干预实验报告.md) | Measured paired response sensitivity and the effect of teacher context and answer reserve. [MATH-500 transfer](results/condition-study/engineering-followup.json) |
| [8B scale and matched-budget reevaluation](8B规模对照报告.md#两种规模的同协议复评) | The relative 1.7B advantage narrowed under some prompts, with a residual gap under matched budget pressure in the earlier clipped recipe. |
| [Retention-aware routing](保留感知蒸馏报告.md) | No-op anchoring kept output tokens identical to the frozen base. The additional gain from behavior-based assignment over matched random routing remained unestablished. |

## References

The OPSD training setup follows [Self-Distilled Reasoner](https://proceedings.mlr.press/v306/zhao26be.html). Related preprints study [correction reversal under vocabulary-wise clipping](https://arxiv.org/abs/2609.38995) and [premature termination and repetition in OPSD](https://arxiv.org/abs/2611.19925). The main evaluation uses [GSM8K](https://github.com/openai/grade-school-math) and [SVAMP](https://aclanthology.org/2021.naacl-main.168/).
