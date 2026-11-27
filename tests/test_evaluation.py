import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from opsdmath.analysis.paired_statistics import interval
from opsdmath.evaluation.answer_parser import audit_final_answer, parse_strict_answer
from opsdmath.evaluation.budget import score
from opsdmath.evaluation.inference import validate_recipe


def test_fixed_prefix_scoring_and_answer_policies():
    class Tokenizer:
        def batch_decode(self, ids, **kwargs):
            return ["".join({1: "Final answer: ", 2: "42"}[token] for token in row) for row in ids]

    predictions = [{"id": "q", "token_ids": [1, 2], "hit_cap": False}]
    labels = {"q": Fraction(42)}
    assert score(predictions, labels, Tokenizer(), 1)[0]["numeric"]["correct"] == 0
    assert score(predictions, labels, Tokenizer(), 2)[0]["strict"]["correct"] == 1
    assert audit_final_answer("Work. Final answer: 3/4") == Fraction(3, 4)
    assert parse_strict_answer("Work. Final answer: 3/4") is None
    estimate = interval(np.array([0, 1, 0, -1]))
    assert estimate["n"] == 4 and estimate["pp"] == 0
    assert estimate["interval95_pp"][0] < 0 < estimate["interval95_pp"][1]


@pytest.mark.parametrize("key, value", [
    ("short", 96), ("long", 1024), ("thinking", True), ("prompt", "legacy"),
    ("batch_size", 0), ("questions", 0),
])
def test_reference_inference_rejects_protocol_drift(key, value):
    path = Path(__file__).resolve().parents[1] / "configs/evaluation_8b.json"
    config = json.loads(path.read_bytes())
    validate_recipe(config)
    with pytest.raises(ValueError, match=key):
        validate_recipe(config | {key: value})
