import json
import tarfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opsdmath.analysis.paired_statistics import interval
from opsdmath.evaluation.answer_parser import audit_final_answer, parse_strict_answer
from opsdmath.evaluation.budget import score
from opsdmath.evaluation import inference
from opsdmath.evaluation.inference import validate_recipe
from opsdmath.io import write_rows
from opsdmath.training import rollout


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


@pytest.mark.parametrize("filename, damage", [
    ("tokens.jsonl", None),
    ("score-128.json", "{}"),
    ("score-512.json", None),
    ("complete.json", "{}"),
    ("artifacts-sha256.json", None),
    ("artifacts-sha256.json", "{"),
    ("protocol.json", None),
    ("protocol.json", "{"),
])
def test_evaluation_regenerates_missing_or_corrupt_outputs(tmp_path, monkeypatch, filename, damage):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [12]

        def batch_decode(self, ids, **kwargs):
            return ["Final answer: 42"] * len(ids)

    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    write_rows(tmp_path / "inputs.jsonl", [{"id": "q", "problem": "40+2?"}])
    write_rows(tmp_path / "labels.jsonl", [{"id": "q", "answer": "42"}])
    args = SimpleNamespace(data=tmp_path, model=model, output=tmp_path / "out", batch_size=1, group=["base"])
    loads = []

    def load_eval_model(path):
        loads.append(path)
        return SimpleNamespace(device="cpu"), Tokenizer()

    monkeypatch.setattr(inference, "load_eval_model", load_eval_model)
    monkeypatch.setattr(rollout, "batch_suffixes", lambda *args: [([12], False)])
    inference.evaluate(args, expected_questions=1)
    inference.evaluate(args, expected_questions=1)
    assert loads == [model]

    group = args.output / "base"
    path = group / filename
    if damage is None:
        path.unlink()
    else:
        path.write_text(damage, encoding="utf-8")
    inference.evaluate(args, expected_questions=1)
    assert loads == [model, model]
    assert inference.outputs_match(group)


def test_svamp_recipe_matches_archived_protocol():
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/evaluation_svamp_8b.json").read_bytes())
    validate_recipe(config)
    with tarfile.open(repo / "results/kl_ablation/controls-evidence.tar.gz") as evidence:
        protocol = json.load(evidence.extractfile("transfer-evaluation/base/protocol.json"))
        metrics = json.load(evidence.extractfile("transfer-evaluation/base/complete.json"))
    for key in ("batch_size", "short", "long", "prompt", "thinking"):
        assert config[key] == protocol[key]
    assert config["model_index_sha256"] == protocol["model"]
    assert config["inputs_sha256"] == protocol["inputs"]
    assert config["labels_sha256"] == protocol["labels"]
    assert config["questions"] == metrics["128"]["numeric"]["n"]
