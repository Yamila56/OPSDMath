# Research archive

This directory preserves the experiments and evidence that preceded the [matched KL ablation](../docs/experiments.md).

The [initial evaluation](results/initial_evaluation) records early scoring and answer-format sensitivity. The [budget study](results/budget_efficiency) records short-generation curves and structured scoring. The [crossed-budget study](results/crossed_budget) compares model sizes under shared prompts and budgets. [Teacher-context probes](results/teacher_context) and [retention studies](results/retention) test privileged information and matched routing.

Run historical experiment modules from the checkout root after an editable install. The launchers use the model environments specified in each script. The [legacy entry points](legacy) preserve the earlier trainer, data and scale-transfer command styles. The official commands and current results are documented in the [reproduction guide](../docs/reproduction.md). Evidence files and the paths recorded inside historical archives retain their original contents.
