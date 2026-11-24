"""Build short-target SFT controls and audit numeric answer delivery."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from gsm8k import number, read_rows, write_rows


EQUATIONS = re.compile(r"<<([^<>]+)>>")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, target):
    rows = read_rows(source)
    if len(rows) != 256 or len({row["id"] for row in rows}) != 256:
        raise ValueError("Expected the original 256 unique training questions")
    targets = []
    fallback_count = 0
    for row in rows:
        if number(row["answer"]) is None:
            raise ValueError(f"Invalid final answer for {row['id']}")
        equations = EQUATIONS.findall(row["solution"]) if target == "equations" else []
        fallback_count += target == "equations" and not equations
        reference = "\n".join([*equations, f"Final answer: {row['answer']}"])
        targets.append({**row, "solution": reference})
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "train.jsonl"
    dataset.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in targets).encode("utf-8"))
    (output / "manifest.json").write_text(json.dumps({
        "source_sha256": sha(source), "train_sha256": sha(dataset),
        "count": len(rows), "target": target, "equation_fallback_count": fallback_count, "epochs": 1,
        "seeds": [17, 29], "updates": 64, "accumulation": 4,
    }, indent=2) + "\n", encoding="utf-8")


def category(short, long, strict):
    if short["parsed"] is not None:
        if not short["correct"]:
            return "parsed_wrong_at_128"
        if not strict["correct"]:
            return "correct_numeric_strict_miss_at_128"
        return "correct_numeric_and_strict_at_128"
    if short["length"] == 128 and long["length"] > 128:
        if long["parsed"] is not None:
            return "first_observed_only_by_512_correct" if long["correct"] else "first_observed_only_by_512_wrong"
    return "unresolved_no_numeric_at_128"


def audit(scores, output):
    summaries = {}
    output.mkdir(parents=True, exist_ok=False)
    reviewed = {}
    for source in sorted(scores.glob("*.jsonl")):
        filtered = [r for r in read_rows(source) if (r["budget"], r["parser"]) in {
            (128, "numeric"), (128, "strict"), (512, "numeric"),
        }]
        index = {(row["id"], row["budget"], row["parser"]):row for row in filtered}
        if len(index) != 1536:
            raise ValueError(f"Expected 512 examples under three conditions for {source}")
        ids = sorted({row["id"] for row in filtered})
        items = []
        for key in ids:
            short = index[key, 128, "numeric"]
            long = index[key, 512, "numeric"]
            strict = index[key, 128, "strict"]
            kind = category(short, long, strict)
            items.append(dict(id=key, category=kind, numeric_128=short["parsed"], numeric_512=long["parsed"],
                              correct_128=short["correct"], correct_512=long["correct"],
                              length_128=short["length"], length_512=long["length"]))
        summaries[source.stem] = dict(Counter(r["category"] for r in items))
        reviewed[source.stem] = {r["id"]: r for r in items}
        write_rows(output / source.name, items)
    pairs = [("online", "sft"), ("answer", "equation"), ("online", "fixed"),
             ("online", "shortsft"), ("answer", "shortsft"),
             ("online", "equationsft"), ("answer", "equationsft")]
    contributions = []
    for left, right in pairs:
        for seed in (17, 29):
            if f"{left}{seed}" not in reviewed or f"{right}{seed}" not in reviewed:
                continue
            aa, bb = reviewed[f"{left}{seed}"], reviewed[f"{right}{seed}"]
            if aa.keys() != bb.keys():
                raise ValueError("Paired ID mismatch")
            groups = Counter()
            for key, a in aa.items():
                b = bb[key]
                left_parsed, right_parsed = a["numeric_128"] is not None, b["numeric_128"] is not None
                group = "both" if left_parsed and right_parsed else "only_left" if left_parsed else (
                    "only_right" if right_parsed else "neither")
                groups[group] += int(a["correct_128"]) - int(b["correct_128"])
            contributions.append(dict(left=left, right=right, seed=seed, **{
                key: groups[key] for key in ("both", "only_left", "only_right", "neither")
            }))
    (output / "counts.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    if contributions:
        with (output / "paired-contributions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contributions[0]))
            writer.writeheader()
            writer.writerows(contributions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", choices=("answer", "equations"), default="answer")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.source, args.output, args.target)
    else:
        audit(args.source, args.output)


if __name__ == "__main__":
    main()
