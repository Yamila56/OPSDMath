"""Prepare reproducible GSM8K splits and score short numeric responses."""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from fractions import Fraction
from pathlib import Path
from urllib.request import urlopen

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SOURCE = f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data"
CHECKSUMS = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
INTEGER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
NUMBER = re.compile(rf"[+-]?(?:{INTEGER}(?:\.[0-9]+|/[0-9]+)?|\.[0-9]+)")
FINAL = re.compile(r"^\s*Final answer\s*:\s*(.*?)\s*$", re.IGNORECASE)
FINAL_MARKER = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+)?\**Final answer\**[ \t]*:[ \t]*\**[ \t]*",
                          re.IGNORECASE | re.MULTILINE)
BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")


def number(text):
    text = text.strip().replace("−", "-")
    if NUMBER.fullmatch(text):
        try:
            return Fraction(text.replace(",", ""))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def parse_strict_answer(response):
    lines = [match[1] for line in response.splitlines() if (match := FINAL.fullmatch(line))]
    return number(lines[-1]) if lines else None


def decorated_number(text):
    text = text.strip().removesuffix(".").strip().strip("*$` ").strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    boxed = BOXED.fullmatch(text)
    if boxed:
        text = boxed[1].strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    return number(text.removeprefix(r"\$").removeprefix("$"))


def parse_answer(response):
    markers = list(FINAL_MARKER.finditer(response))
    if markers:
        tail = response[markers[-1].end():]
        lines = [line.strip() for line in tail.splitlines() if line.strip(" \t$")]
        if not lines or len(list(BOXED.finditer(tail))) > 1:
            return None
        if any(decorated_number(line) is not None for line in lines[1:]):
            return None
        return decorated_number(lines[0])
    boxes = list(BOXED.finditer(response))
    if len(boxes) == 1 and not response[boxes[0].end():].strip(" \r\n\t$.*\\)]"):
        return decorated_number(boxes[0][1])
    return None


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def dataset(split, cache):
    path = cache / f"{split}.jsonl"
    if path.exists():
        payload = path.read_bytes()
    else:
        with urlopen(f"{SOURCE}/{split}.jsonl", timeout=30) as stream:
            payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != CHECKSUMS[split]:
        raise ValueError(f"Unexpected GSM8K checksum for {split}")
    path.write_bytes(payload)
    unique = {}
    for line in payload.splitlines():
        row = json.loads(line)
        normalized = " ".join(unicodedata.normalize("NFKC", row["question"]).lower().split())
        key = hashlib.sha256(f"opsd-v1|{normalized}".encode()).hexdigest()
        rationale, label = row["answer"].rsplit("####", 1)
        label = label.strip()
        if number(label) is None:
            raise ValueError(f"Invalid GSM8K gold answer at {split}/{key}")
        previous = unique.setdefault(key, {
            "id": key, "problem": row["question"], "solution": f"{rationale.strip()}\nFinal answer: {label}",
            "answer": label,
        })
        if number(label) != number(previous["answer"]):
            raise ValueError(f"Conflicting duplicate at {split}/{key}")
    return unique, len(payload.splitlines())


def prepare(output):
    cache = output / "source"
    cache.mkdir(parents=True, exist_ok=True)
    train, raw_train = dataset("train", cache)
    test, raw_test = dataset("test", cache)
    overlap = train.keys() & test.keys()
    train_sorted = [train[key] for key in sorted(train.keys() - overlap)]
    test_sorted = [test[key] for key in sorted(test)]
    splits = {"dev": train_sorted[:128], "train": train_sorted[128:384], "test": test_sorted[:256]}
    for split, rows in splits.items():
        if split == "test":
            write_rows(output / "test_inputs.jsonl", [{k: row[k] for k in ("id", "problem")} for row in rows])
            write_rows(output / "test_labels.jsonl", [{k: row[k] for k in ("id", "answer")} for row in rows])
        else:
            write_rows(output / f"{split}.jsonl", rows)
    manifest = {
        "source_revision": REVISION, "source_sha256": CHECKSUMS,
        "source_counts": {"train": raw_train, "test": raw_test},
        "within_split_duplicates": {"train": raw_train - len(train), "test": raw_test - len(test)},
        "train_test_overlap_removed": len(overlap),
        "quick_counts": {"train": 64, "test": 96},
        "counts": {split: len(rows) for split, rows in splits.items()},
        "file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.jsonl"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def score(labels_path, predictions_paths, limit):
    gold = {row["id"]: number(row["answer"]) for row in read_rows(labels_path)[:limit]}
    if not gold or None in gold.values():
        raise ValueError("Gold split must contain valid answers")
    outcomes = {}
    metrics = {}
    for path in predictions_paths:
        rows = read_rows(path)
        actual = {row["id"]: parse_answer(row["response"]) for row in rows}
        if len(actual) != len(rows) or actual.keys() != gold.keys():
            raise ValueError(f"Prediction IDs do not match gold exactly: {path}")
        valid = sum(value is not None for value in actual.values())
        strict = {row["id"]: parse_strict_answer(row["response"]) for row in rows}
        outcomes[path.stem] = [int(actual[key] == answer) for key, answer in gold.items()]
        correct = sum(outcomes[path.stem])
        metrics[path.stem] = {
            "n": len(gold), "correct": correct, "accuracy": correct / len(gold),
            "valid": valid, "valid_accuracy": correct / valid if valid else None,
            "strict_valid": sum(value is not None for value in strict.values()),
            "strict_correct": sum(strict[key] == answer for key, answer in gold.items()),
            "token_cap_hits": sum(row["hit_cap"] for row in rows),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in rows) / len(gold),
        }
    comparisons = {}
    for comparator in ("sft", "base"):
        if "opsd" in outcomes and comparator in outcomes:
            pairs = list(zip(outcomes["opsd"], outcomes[comparator], strict=True))
            deltas = [op - other for op, other in pairs]
            rng = random.Random(17)
            draws = sorted(sum(rng.choices(deltas, k=len(pairs))) / len(pairs) for _ in range(10000))
            comparisons[f"opsd_minus_{comparator}"] = {
                "delta": sum(deltas) / len(deltas), "ci95": [draws[250], draws[9749]],
                "only_opsd_correct": deltas.count(1), "only_comparator_correct": deltas.count(-1),
                "both_correct": pairs.count((1, 1)), "both_wrong": pairs.count((0, 0)),
            }
    return {"metrics": metrics, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("output", type=Path)
    evaluate = subparsers.add_parser("score")
    evaluate.add_argument("labels", type=Path)
    evaluate.add_argument("predictions", type=Path, nargs="+")
    evaluate.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    else:
        print(json.dumps(score(args.labels, args.predictions, args.limit), indent=2))


if __name__ == "__main__":
    main()
