"""Verify the SVAMP transfer comparison and track the frozen model's early answers."""

import numpy as np

from opsdmath.analysis.paired_statistics import interval
from opsdmath.evaluation.metrics import LONG, SHORT, cohort, read_correctness


def summarize(folder):
    strata = cohort(folder / "base")
    early = {key for key, category in strata.items() if category == "early"}
    late = {key for key, category in strata.items() if category == "late"}
    result = {"accuracy": {}, "early_retention": {}, "late_acceleration": {}}
    for budget in (SHORT, LONG):
        base = read_correctness(folder / "base", budget)
        models = {}
        for seed in ("17", "29"):
            learner = read_correctness(folder / f"full-{seed}", budget)
            if learner.keys() != base.keys():
                raise ValueError(f"SVAMP seed {seed} uses a different question set at {budget} tokens")
            difference = np.array([int(learner[key]) - int(base[key]) for key in base])
            models[seed] = {"correct": sum(learner.values()), "vs_base": interval(difference)}
            if budget == SHORT:
                result["early_retention"][seed] = {"correct": sum(learner[key] for key in early), "n": len(early)}
                result["late_acceleration"][seed] = {"correct": sum(learner[key] for key in late), "n": len(late)}
        result["accuracy"][str(budget)] = {"base": sum(base.values()), "n": len(base), "models": models}
    return result
