# Supplementary studies

The earlier experiments tested whether budget guidance, answer format, teacher reference information, and training-example selection changed short-budget outcomes. Their dataset splits and protocols differ from the matched Qwen3-8B KL ablation.

| Question | Code | Evidence |
| --- | --- | --- |
| How do shorter budgets and structured answer fields affect scoring? | [Budget and format studies](../archive/experiments/budget_efficiency) | [Budget curves and paired scores](../archive/results/budget_efficiency) |
| How sensitive is the comparison to matched budget pressure and prompts across model sizes? | [Cross-scale sweep](../archive/experiments/budget_efficiency/summarize_scale_budget_crossed.py) | [Crossed-budget evaluation](../archive/results/crossed_budget) |
| Which forms of privileged context support answer delivery? | [Teacher context probes](../archive/experiments/teacher_context) | [Context and format comparisons](../archive/results/teacher_context) |
| Does behavior-based routing beat matched random routing? | [Retention experiments](../archive/experiments/retention) | [Retention and continuation studies](../archive/results/retention) |
| Can identical checkpoints reproduce the same optimizer update and rollout sequence? | [Trainer validation](trainer_parity.md) | [Pinned upstream audit](../results/trainer_validation) |

The routing runs verified a token-identical no-op anchor. The measured advantage over matched random routing remained uncertain. The [archive index](../archive/README.md) links to the preserved early results and compatibility entry points.
