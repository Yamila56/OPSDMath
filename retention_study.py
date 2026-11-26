"""Freeze data and measure retention and acceleration under a shared budget prompt."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from eval_scale_transfer import digest, score, validate_rows
from gsm8k import REVISION, dataset, number, read_rows, write_rows


SHORT, LONG = 128, 512
SEEDS = (17, 29)
STRATA = ("early", "late", "wrong_both")


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_split(folder, rows):
    folder.mkdir()
    for name, fields in (("inputs", ("id", "problem")), ("labels", ("id", "answer"))):
        write_rows(folder / f"{name}.jsonl", [{k: r[k] for k in fields} for r in rows])


def prepare(args):
    from short_sft_baseline import prepare as prepare_equations

    train = read_rows(args.train)
    original_train, _ = dataset("train", args.source)
    original_test, _ = dataset("test", args.source)
    reviewed = {r["id"] for path in args.reviewed for r in read_rows(path)}
    train_ids = {r["id"] for r in train}
    if len(train_ids) != len(train) or not train_ids <= original_train.keys() - original_test.keys():
        raise ValueError("The training questions must be unique and from the original training source")
    test_ids = sorted(original_test.keys() - original_train.keys() - reviewed)
    if not test_ids:
        raise ValueError("No independent test questions remain")
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output / "train.jsonl", train)
    prepare_equations(args.train, args.output / "equations", "equations")
    write_split(args.output / "training-cohort", train)
    write_split(args.output / "test", [original_test[x] for x in test_ids])
    write(args.output / "data-manifest.json", dict(
        revision=REVISION, excluded_reviewed=len(reviewed), train=len(train_ids), test=len(test_ids),
        sha256={str(p.relative_to(args.output).as_posix()): digest(p)
                for p in sorted(args.output.rglob("*.jsonl"))},
    ))


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


def route(args):
    grouping = cohort(args.base)
    source_order = [r["id"] for r in read_rows(args.train)]
    if len(source_order) != len(grouping) or grouping.keys() != set(source_order):
        raise ValueError("Routing requires frozen predictions on the exact training IDs")
    report = Counter(grouping.values())
    n_answer = report["late"]
    if not 0 < n_answer < len(grouping):
        raise ValueError("Training cohorts do not support a selective comparison")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "training-cohorts.json", grouping)
    write(args.output / "cohort-counts.json", report)
    write(args.output / "behavior.json", {k: "answer" if v == "late" else "none" for k, v in grouping.items()})
    ordered = sorted(grouping)
    schedule = {}
    for seed in SEEDS:
        # Separate the random assignment stream from the trainer's shuffle.
        sampled = set(random.Random(f"opsd-teacher-assignment:{seed}").sample(ordered, n_answer))
        write(args.output / f"random-{seed}.json", {k: "answer" if k in sampled else "none" for k in ordered})
        encounter_order = source_order.copy()
        random.Random(seed).shuffle(encounter_order)
        schedule[str(seed)] = [i for i, key in enumerate(encounter_order, 1) if key in sampled]
    write(args.output / "random-training-positions.json", schedule)


def evaluate(args):
    from peft import PeftModel

    from opsd_calibration import load_model
    from run_math import batch_suffixes, prompt

    n = len(read_rows(args.data / "inputs.jsonl"))
    rows, labels = validate_rows(args.data / "inputs.jsonl", args.data / "labels.jsonl", n)
    truth = {row["id"]: number(row["answer"]) for row in labels}
    protocol = dict(batch_size=args.batch_size, short=SHORT, long=LONG,
                    model=digest(args.model / "model.safetensors.index.json"),
                    inputs=digest(args.data / "inputs.jsonl"), labels=digest(args.data / "labels.jsonl"),
                    prompt="combined-128", thinking=False)
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in args.group:
        name, _, adapter = spec.partition("=")
        output = args.output / name
        identity = dict(protocol, adapter=digest(Path(adapter) / "adapter_model.safetensors") if adapter else None)
        output.mkdir(exist_ok=True)
        manifest = output / "protocol.json"
        if manifest.exists() and identity != json.loads(manifest.read_text(encoding="utf-8")):
            raise ValueError(f"Evaluation inputs changed for {name}")
        if (output / "complete.json").exists():
            continue
        write(manifest, identity)
        base, tokenizer = load_model(args.model)
        model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base
        traces = []
        for start in range(0, n, args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [prompt(tokenizer, r, output_budget=SHORT) for r in batch]
            generated = batch_suffixes(model, tokenizer, prompts, LONG, False, model.device)
            if start == 0:
                direct = batch_suffixes(model, tokenizer, prompts, SHORT, False, model.device)
                if any(a[:SHORT] != b for (a, _), (b, _) in zip(generated, direct, strict=True)):
                    raise ValueError("Direct short decoding differs from the long trajectory prefix")
            traces.extend(dict(id=row["id"], token_ids=ids, hit_cap=hit)
                          for row, (ids, hit) in zip(batch, generated, strict=True))
        temporary = output / "tokens.tmp"
        write_rows(temporary, traces)
        temporary.replace(output / "tokens.jsonl")
        metrics = {}
        for cap in (SHORT, LONG):
            metrics[str(cap)], per_item = score(traces, truth, tokenizer, cap)
            write(output / f"score-{cap}.json", per_item)
        write(output / "complete.json", metrics)
        print(name, metrics["128"]["numeric"], flush=True)
        del model, base


def summarize(args):
    import numpy as np

    grouping = cohort(args.evaluation / "base")
    ids = list(grouping)
    masks = {key: np.array([grouping[x] == key for x in ids]) for key in STRATA}
    methods, evaluation = {}, {}
    for path in sorted(args.evaluation.iterdir()):
        if not (path / "complete.json").exists():
            continue
        rows = json.loads((path / "score-128.json").read_text(encoding="utf-8"))["numeric"]
        index = {r["id"]: r["correct"] for r in rows}
        if len(rows) != len(ids) or index.keys() != grouping.keys():
            raise ValueError(f"Nonmatching test IDs for {path.name}")
        methods[path.name] = np.array([index[x] for x in ids], dtype=np.int8)
        evaluation[path.name] = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    results = {name: dict(correct=int(values.sum()), by_cohort={key: int(values[mask].sum())
                    for key, mask in masks.items()}, budgets=evaluation[name]) for name, values in methods.items()}
    rng = np.random.default_rng(62026)
    differences = {}
    for seed in SEEDS:
        selective = f"behavior-{seed}"
        for control in ("privileged", "random", "equations", "unprivileged", "base"):
            comparator = control if control == "base" else f"{control}-{seed}"
            if selective not in methods or comparator not in methods:
                continue
            differences[control, seed] = methods[selective] - methods[comparator]
    for control in sorted({key[0] for key in differences}):
        if all((control, seed) in differences for seed in SEEDS):
            differences[control, "seed_mean"] = np.mean([differences[control, seed] for seed in SEEDS], axis=0)
    contrasts = {}
    for (control, seed), delta in differences.items():
        relative = {}
        for key, values in (("all", delta), *((key, delta[mask]) for key, mask in masks.items())):
            if values.size:
                draw = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
                relative[key] = dict(delta=float(values.mean()), ci95=np.quantile(draw, [.025, .975]).tolist())
        contrasts[f"behavior_vs_{control}_{seed}"] = relative
    write(args.output, dict(n=len(ids), denominators=Counter(grouping.values()), results=results, contrasts=contrasts))


def training_manifest(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source_ids = [r["id"] for r in read_rows(args.train)]
    group_summaries = {}
    for spec in args.group:
        name, _, directory = spec.partition("=")
        folder = Path(directory)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        events = read_rows(folder / "events.jsonl")
        steps = [event for event in events if event["event"] == "update"]
        examples = [event for event in events if event["event"] == "microstep"]
        batches = [event for event in events if event["event"] == "rollout_batch"]
        data_seed = config["data_seed"] if config["data_seed"] is not None else config["seed"]
        order = source_ids.copy()
        random.Random(data_seed).shuffle(order)
        if [r["id"] for r in examples] != order or events[-1]["event"] != "complete":
            raise ValueError(f"Training ID order or completion mismatch for {name}")
        routing = config.get("teacher_by_id") or {}
        start, learning_rate = 0, config["learning_rate"]
        results = []
        counts = Counter()
        for index, update in enumerate(steps, 1):
            end = update["microsteps"]
            rows = examples[start:end]
            listed = []
            for row in rows:
                expected = routing.get(row["id"], config["teacher_context"])
                branch = row.get("teacher_context", expected) if config["mode"] == "opsd" else "sft"
                if config["mode"] == "opsd" and branch != expected:
                    raise ValueError(f"Teacher assignment mismatch in {name} at {row['id']}")
                counts[branch] += 1
                listed.append({k: row.get(k) for k in ("id", "tokens", "exact_logits_match", "unclipped_kl",
                                                        "clipped_kl", "hit_cap")} | {"branch": branch})
            tokens = batches[index - 1]["tokens"] if batches else None
            if tokens is not None and tokens != sum(row["tokens"] for row in rows):
                raise ValueError(f"Rollout token count mismatch in {name}, update {index}")
            results.append(dict(update=index, lr_used=learning_rate, lr_after=update["learning_rate"],
                                grad_norm_before_clip=update["grad_norm"], rollout_tokens=tokens, questions=listed))
            start, learning_rate = end, update["learning_rate"]
        if start != len(order) or (batches and len(batches) != len(steps)):
            raise ValueError(f"Update count mismatch for {name}")
        write_rows(output / f"{name}.jsonl", results)
        group_summaries[name] = dict(
            updates=len(results), branches=counts, rollout_tokens=sum(x["tokens"] for x in batches),
            seconds=events[-1]["seconds"], init_seed=config["init_seed"] if config["init_seed"] is not None else config["seed"],
            data_seed=data_seed, rollout_seed=config["rollout_seed"],
            rollout_rng="continuous torch RNG after initialization" if config["rollout_seed"] is None else "explicit",
            routing_rng=f"opsd-teacher-assignment:{config['seed']}" if name.startswith("random-") else None,
            eval_seed=None, evaluation="deterministic greedy generation",
            ordered_ids_sha256=hashlib.sha256("\n".join(order).encode()).hexdigest(),
        )
    write(output / "summary.json", group_summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, action, fields in (("prepare", prepare, ("train", "source", "output")),
                                 ("route", route, ("base", "train", "output")),
                                 ("evaluate", evaluate, ("model", "data", "output")),
                                 ("summarize", summarize, ("evaluation", "output")),
                                 ("training-manifest", training_manifest, ("train", "output"))):
        cmd = sub.add_parser(name)
        cmd.set_defaults(action=action)
        for field in fields:
            cmd.add_argument(f"--{field}", type=Path, required=True)
        if name == "prepare":
            cmd.add_argument("--reviewed", nargs="+", type=Path, required=True)
        elif name == "training-manifest":
            cmd.add_argument("--group", nargs="+", required=True)
        elif name == "evaluate":
            cmd.add_argument("--group", nargs="+", required=True)
            cmd.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.action(args)
