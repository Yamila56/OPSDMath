"""Freeze the confirmation split and score saved generation prefixes."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from opsdmath.analysis.paired_statistics import uncertainty
from opsdmath.data.gsm8k import dataset, number, parse_answer, parse_strict_answer, read_rows, write_rows
from archive.experiments.budget_efficiency.summarize_run import audit_final_answer, write_csv

BUDGETS = (64, 128, 256, 512)
PARSERS = {"numeric": audit_final_answer, "original": parse_answer, "strict": parse_strict_answer}


def prepare(data, output):
    train, _ = dataset("train", data / "source")
    test, _ = dataset("test", data / "source")
    old = {row["id"] for row in read_rows(data / "test_labels.jsonl")}
    ids = sorted(test.keys() - train.keys() - old)[:512]
    if len(ids) != 512:
        raise ValueError("Expected 512 independent confirmation IDs")
    output.mkdir(parents=True, exist_ok=False)
    for stem, fields in (("inputs", ["id", "problem"]), ("labels", ["id", "answer", "solution"])):
        write_rows(output / f"{stem}.jsonl", [{key: test[i][key] for key in fields} for i in ids])
    operation_counts = Counter(row["solution"].count("<<") for row in read_rows(data / "train.jsonl"))
    manifest = {"n": len(ids), "rule": "first 512 sorted IDs outside source train and old test",
                "train_equation_counts": sorted(operation_counts.items()),
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def report(model, labels, output, groups, *, budgets=BUDGETS):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    generation = json.loads((model / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    label_rows = read_rows(labels)
    truth = {r["id"]: number(r["answer"]) for r in label_rows}
    steps = {r["id"]: r.get("solution", "").count("<<") if "solution" in r else None for r in label_rows}
    values = {}
    metrics = []
    comparison = []
    output.mkdir(parents=True, exist_ok=False)
    for name, src in (g.split("=", 1) for g in groups):
        source = read_rows(Path(src))
        if {r["id"] for r in source} != truth.keys() or len(source) != len(truth):
            raise ValueError(f"Unexpected prediction IDs for {name}")
        records = []
        for budget in budgets:
            decoded = tokenizer.batch_decode([r["token_ids"][:budget] for r in source], skip_special_tokens=True)
            for parser_name, parse in PARSERS.items():
                vectors, lengths, caps = {}, [], 0
                for row, response in zip(source, decoded, strict=True):
                    key = row["id"]
                    suffix = row["token_ids"][:budget]
                    answer = parse(response)
                    vectors[key] = (answer == truth[key], answer is not None)
                    lengths.append(len(suffix))
                    caps += len(suffix) == budget and suffix[-1] not in eos
                    records.append(dict(id=key, budget=budget, parser=parser_name, text=response,
                                        length=len(suffix), parsed=str(answer) if answer is not None else None,
                                        correct=answer == truth[key], operation_count=steps[key]))
                values[name, budget, parser_name] = vectors
                correct = sum(x[0] for x in vectors.values())
                parsed = sum(x[1] for x in vectors.values())
                metrics.append(dict(method=name, budget=budget, parser=parser_name, n=len(source),
                    correct=correct, accuracy=correct/len(source), parsed=parsed, invalid=len(source)-parsed,
                    parsed_wrong=parsed-correct, mean_tokens=mean(lengths), median_tokens=median(lengths), cap_hits=caps))
        write_rows(output / f"{name}.jsonl", records)
    comparisons = (
        ("equation", "answer"), ("equation", "online"), ("online", "fixed"), ("online", "sft"),
        ("online", "shortsft"), ("answer", "shortsft"), ("equation", "shortsft"), ("sft", "shortsft"),
        ("online", "equationsft"), ("answer", "equationsft"), ("equation", "equationsft"),
        ("sft", "equationsft"), ("equationsft", "shortsft"),
    )
    for left, right in comparisons:
        for seed in (17, 29):
            for budget in budgets:
                for parser in PARSERS:
                    a = values.get((f"{left}{seed}", budget, parser))
                    b = values.get((f"{right}{seed}", budget, parser))
                    if a is None or b is None:
                        continue
                    diff = [int(a[k][0])-int(b[k][0]) for k in truth]
                    paired = [d for k, d in zip(truth, diff, strict=True) if a[k][1] and b[k][1]]
                    low, high = uncertainty(diff)
                    comparison.append(dict(left=left, right=right, seed=seed, budget=budget, parser=parser,
                        delta=mean(diff), low=low, high=high, common_parsed=len(paired),
                        common_delta=mean(paired) if paired else None))
    write_csv(output / "metrics.csv", metrics)
    if comparison:
        write_csv(output / "comparisons.csv", comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("data", type=Path)
    prep.add_argument("output", type=Path)
    score = subs.add_parser("report")
    score.add_argument("model", type=Path)
    score.add_argument("labels", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("groups", nargs="+")
    score.add_argument("--budgets", nargs="+", type=int, choices=BUDGETS, default=BUDGETS)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.data, args.output)
    else:
        report(args.model, args.labels, args.output, args.groups, budgets=args.budgets)


if __name__ == "__main__":
    main()
