"""Compare privileged distillation gradients on frozen reference and student trajectories."""

import argparse
import ast
import operator
from collections import defaultdict
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from statistics import mean

from gsm8k import number, read_rows, write_rows
from opsd_calibration import load_model, write
from short_sft_baseline import EQUATIONS

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate_expression(text):
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return walk(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Unsupported equation")

    return walk(ast.parse(text.replace(",", "").replace("×", "*").strip(), mode="eval").body)


def compact_reference(row):
    lines = EQUATIONS.findall(row["solution"])
    if not lines:
        return None
    try:
        answers = []
        for line in lines:
            left, right = line.split("=")
            value = evaluate_expression(right)
            if evaluate_expression(left) != value:
                return None
            answers.append(value)
        if answers[-1] != number(row["answer"]):
            return None
    except (ValueError, SyntaxError, ArithmeticError):
        return None
    return "\n".join(lines), f"\nFinal answer: {row['answer']}"


def gradients(student, teacher, temperature, clip):
    p_log = (student.float() / temperature).log_softmax(-1)
    q_log = (teacher.float() / temperature).log_softmax(-1)
    p, q = p_log.exp(), q_log.exp()
    capped = q * (q_log-p_log) > clip
    active = q.masked_fill(capped, 0)
    reverse = p_log-q_log
    return p, {
        "clipped": (p*active.sum(-1, keepdim=True)-active)/temperature,
        "forward": (p-q)/temperature,
        "reverse": p*(reverse-(p*reverse).sum(-1, keepdim=True))/temperature,
    }, (q*capped).sum(-1)


def unit_cos(a, b):
    norm = float(a.norm()*b.norm())
    return float((a*b).sum())/norm if norm else None


def audit(args):
    import torch

    from run_math import DEFAULT_KL_CLIP, ROLLOUT_TEMPERATURE, batch_suffixes, completion_logits, distillation_loss, prompt

    check_student = torch.tensor([[0., 1., 2., 3.]], requires_grad=True)
    check_teacher = torch.tensor([[3., 0., 2., 1.]])
    _, check, _ = gradients(check_student, check_teacher, ROLLOUT_TEMPERATURE, DEFAULT_KL_CLIP)
    actual, = torch.autograd.grad(distillation_loss(check_student, check_teacher), check_student)
    torch.testing.assert_close(actual, check["clipped"], atol=1e-7, rtol=1e-5)

    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_rows(args.dev)
    reference = {row["id"]: compact_reference(row) for row in rows}
    results = []
    coverage = {}
    models = {"1.7B":args.small, "8B":args.large}
    arms = [(scale, scale, None) for scale in models]
    if args.adapter:
        from peft import PeftModel

        arms = [(f"{scale}/{seed}", scale, Path(path)) for scale, seed, path in args.adapter]
    for name, scale, adapter in arms:
        base, tokenizer = load_model(models[scale])
        model = base if adapter is None else PeftModel.from_pretrained(base, adapter).eval()
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        train_refs = [compact_reference(row) for row in read_rows(args.train)]
        lengths = [len(encode("".join(parts)))+1 for parts in train_refs if parts is not None]
        coverage[scale] = dict(total=len(train_refs), arithmetic_checked=len(lengths),
                               within_128=sum(n <= 128 for n in lengths), max_tokens=max(lengths))
        with torch.inference_mode():
            for row in rows:
                ident = row["id"]
                student = prompt(tokenizer, row, output_budget=128)
                teacher = prompt(tokenizer, row, privileged=True, context="answer", output_budget=128)
                online, _ = batch_suffixes(model, tokenizer, [student], 33, False, model.device)[0]
                trajectories = [("student", online, ["trajectory"]*len(online))]
                parts = reference[ident]
                if parts:
                    math, answer = [encode(part) for part in parts]
                    all_ids = math + answer + [tokenizer.eos_token_id]
                    if len(all_ids) <= 128:
                        trajectories.append(("verified_compact", all_ids,
                                             ["equation"]*len(math)+["answer"]*len(answer)+["eos"]))
                for source, ids, kinds in trajectories:
                    suffix = torch.tensor([ids], device=model.device)
                    p_logits = completion_logits(model, student.to(model.device), suffix)[0]
                    with model.disable_adapter() if adapter else nullcontext():
                        q_logits = completion_logits(model, teacher.to(model.device), suffix)[0]
                    _, objectives, capped_mass = gradients(p_logits, q_logits, ROLLOUT_TEMPERATURE, DEFAULT_KL_CLIP)
                    ce = p_logits.float().softmax(-1)
                    ce.scatter_add_(1, suffix.T, torch.full_like(suffix.T, -1, dtype=ce.dtype))
                    eos = tokenizer.eos_token_id
                    for i, (token, kind) in enumerate(zip(ids, kinds, strict=True)):
                        if source == "student" and i not in (0, 16, 32):
                            continue
                        g = {name: gradient[i] for name, gradient in objectives.items()}
                        # Infinitesimal gradient descent effect on log probability of a chosen token.
                        raw_p = p_logits[i].float().softmax(-1)
                        effects = {name: float(raw_p @ v-v[token]) for name, v in g.items()}
                        eos_effects = {name: float(raw_p @ v-v[eos]) for name, v in g.items()}
                        results.append(dict(id=ident, scale=name, source=source, position=i, kind=kind,
                                            clipped_mass=float(capped_mass[i]), clip_forward_cos=unit_cos(g["clipped"],g["forward"]),
                                            ce_cos={name: unit_cos(v,ce[i]) for name,v in g.items()} if parts and source=="verified_compact" else None,
                                            target_logprob_direction=effects, eos_logprob_direction=eos_effects))
        del model, base
        torch.cuda.empty_cache()
        print(scale, "completed", flush=True)
    write_rows(args.output/"positions.jsonl", results)
    grouped = defaultdict(list)
    for item in results:
        category = item["kind"] if item["source"]=="verified_compact" else f"student/{item['position']}"
        grouped[item["scale"], category].append(item)
    summary = {}
    for (scale, kind), group in grouped.items():
        cosines = [r["clip_forward_cos"] for r in group if r["clip_forward_cos"] is not None]
        per_q = defaultdict(list)
        for row in group:
            per_q[row["id"]].append(row["clipped_mass"])
        summary[f"{scale}/{kind}"] = dict(
            questions=len(per_q), positions=len(group), clipped_positions=sum(r["clipped_mass"]>0 for r in group),
            mean_clipped_teacher_mass=mean(map(mean, per_q.values())),
            mean_clip_forward_cos=mean(cosines) if cosines else None,
            negative_clip_forward=sum(x<0 for x in cosines),
            ce_cos={name: mean(valid) if (valid:=[r["ce_cos"][name] for r in group if r["ce_cos"] and r["ce_cos"][name] is not None]) else None
                    for name in ("clipped","forward","reverse")},
            target_direction_reversals=sum(r["target_logprob_direction"]["clipped"]*r["target_logprob_direction"]["forward"]<0 for r in group),
            eos_direction_reversals=sum(r["eos_logprob_direction"]["clipped"]*r["eos_logprob_direction"]["forward"]<0 for r in group),
        )
    write(args.output/"summary.json", dict(training_reference_coverage=coverage, results=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("small", "large", "dev", "train", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--adapter", action="append", nargs=3, metavar=("SCALE", "SEED", "PATH"))
    audit(parser.parse_args())
