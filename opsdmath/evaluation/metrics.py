"""Baseline strata and paired question-level bootstrap."""

import json

import numpy as np

SHORT = 128
LONG = 512


def cohort(base):
    def read(cap):
        results = json.loads((base / f"score-{cap}.json").read_text(encoding="utf-8"))["numeric"]
        indexed = {r["id"]: r["correct"] for r in results}
        if len(indexed) != len(results):
            raise ValueError("Repeated prediction IDs")
        return indexed

    short, long = read(SHORT), read(LONG)
    if short.keys() != long.keys():
        raise ValueError("Short and long evaluation IDs differ")
    return {key: "early" if value else "late" if long[key] else "wrong_both" for key, value in short.items()}


def interval(values):
    if not len(values):
        return dict(n=0, pp=None, interval95_pp=None)
    rng = np.random.default_rng(84081)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return dict(n=len(values), pp=float(values.mean()*100), interval95_pp=(np.quantile(draws, [.025, .975])*100).tolist())
