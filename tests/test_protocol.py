from argparse import Namespace
from fractions import Fraction

import numpy as np
import pytest
import torch

from opsdmath.analysis.gradients import gradients
from opsdmath.evaluation.answer_parser import audit_final_answer, parse_strict_answer
from opsdmath.evaluation.budget import score
from opsdmath.evaluation.metrics import interval
from opsdmath.objectives.kl import DEFAULT_KL_CLIP, ROLLOUT_TEMPERATURE, check_loss, distillation_loss
from opsdmath.training.arguments import reference_arguments
from opsdmath.training.checkpoint import save_state, training_identity, validate_resume
from opsdmath.training.prompts import prompt


def test_forward_clipped_and_exact_zero():
    check_loss()


def test_analytic_clipped_and_forward_gradients():
    student = torch.tensor([[0., 1., 2., 3.]], requires_grad=True)
    teacher = torch.tensor([[3., 1., 1., 0.]])
    _, analytic, _ = gradients(student, teacher, ROLLOUT_TEMPERATURE, DEFAULT_KL_CLIP)
    for name, cap in (("clipped", DEFAULT_KL_CLIP), ("forward", None)):
        actual, = torch.autograd.grad(distillation_loss(student, teacher, kl_clip=cap), student)
        torch.testing.assert_close(actual, analytic[name])


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


def test_privileged_prompt_on_same_budget():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"]

    item = {"problem": "2 + 3?", "answer": "5"}
    student = prompt(Tokenizer(), item, output_budget=128)
    teacher = prompt(Tokenizer(), item, True, "answer", output_budget=128)
    assert "Reference Solution Begin" not in student
    assert "Final answer: 5" in teacher
    guidance = "The complete response must fit within 128 tokens. Prioritize the key steps."
    assert guidance in student and guidance in teacher


def test_checkpoint_save_and_resume_identity(tmp_path):
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/opsd_8b_full_kl.json").read_bytes())
    args = Namespace(**(cfg["arguments"] | cfg["seeds"]["17"] | {
        "seed": 17, "mode": "opsd", "model": "model", "teacher_by_id": None,
    }))
    first, second = {"id": "a"}, {"id": "b"}
    identity = training_identity(args, [first, second])
    assert identity != training_identity(args, [second, first])
    fixed = training_identity(Namespace(**(vars(args) | {"fixed_rollout": True})), [first, second])
    assert fixed["fixed_rollout"] is True
    assert identity["fixed_rollout"] is False
    with pytest.raises(ValueError, match="original model"):
        validate_resume(fixed, identity, tmp_path / "train-state.pt")

    old = {key: value for key, value in identity.items() if key not in ("device", "fixed_rollout")}
    (tmp_path / "config.json").write_text(json.dumps({"fixed_rollout": False, "device": "cuda"}))
    validate_resume(old, identity, tmp_path / "train-state.pt")
    (tmp_path / "config.json").write_text(json.dumps({"fixed_rollout": True, "device": "cuda"}))
    with pytest.raises(ValueError, match="original model"):
        validate_resume(old, identity, tmp_path / "train-state.pt")

    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    parameter.backward()
    optimizer.step()
    scheduler.step()
    save_state(tmp_path, "checkpoint-001", identity, 4, 1, optimizer, scheduler, torch.device("cpu"))
    state = torch.load(tmp_path / "train-state.pt", weights_only=True)
    assert state["identity"] == identity and state["adapter"] == "checkpoint-001"
    assert state["next_index"] == 4 and state["update"] == 1
    assert state["cpu_rng"].equal(torch.get_rng_state())
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize("recipe", ["opsd_8b_full_kl", "opsd_8b_clipped_kl", "equation_sft_8b"])
def test_recipe_matches_trainer(recipe):
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / f"configs/{recipe}.json"
    config = json.loads(path.read_bytes())
    for seed in (17, 29):
        paths = Namespace(model="model", data="data", output="out", seed=seed)
        overrides = config["seeds"][str(seed)]
        command = reference_arguments(config, overrides, paths, 256)
        assert command[0] == config["mode"]
        if "kl_clip" in config["arguments"]:
            clip = config["arguments"]["kl_clip"]
            assert command[command.index("--kl-clip") + 1] == ("none" if clip is None else str(clip))
    with pytest.raises(ValueError, match="Recorded updates"):
        reference_arguments(config | {"updates": 65}, overrides, paths, 256)
    if config["mode"] == "opsd":
        with pytest.raises(ValueError, match="Recorded rollout_top_p"):
            reference_arguments(config | {"rollout_top_p": .8}, overrides, paths, 256)
        with pytest.raises(ValueError, match="does not match"):
            bad = config | {"arguments": config["arguments"] | {"unused_flag": False}}
            reference_arguments(bad, overrides, paths, 256)
