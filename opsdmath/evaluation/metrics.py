"""Baseline strata and paired question-level bootstrap."""

import json

SHORT = 128
LONG = 512


def read_correctness(folder, cap, scorer="numeric"):
    path = folder / f"score-{cap}.json"
    results = json.loads(path.read_text(encoding="utf-8"))[scorer]
    indexed = {row["id"]: row["correct"] for row in results}
    if len(indexed) != len(results):
        raise ValueError(f"Repeated prediction IDs in {path}")
    return indexed


def cohort(base, short_budget=SHORT, long_budget=LONG):
    short = read_correctness(base, short_budget)
    long = read_correctness(base, long_budget)
    if short.keys() != long.keys():
        raise ValueError("Short and long evaluation IDs differ")
    return {key: "early" if value else "late" if long[key] else "wrong_both" for key, value in short.items()}
