"""Select anonymized answer, formatting and cutoff cases from the confirmation split."""
import argparse
import json
import random
from pathlib import Path

from opsdmath.data.gsm8k import read_rows, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    truth = {row["id"]: row["answer"] for row in read_rows(args.labels)}
    rng = random.Random(29)
    cases, keys = [], []
    for seed in (17, 29):
        files = (args.scores / f"{group}{seed}.jsonl" for group in ("equation", "answer"))
        a, b = [{(r["id"], r["parser"]): r for r in read_rows(path) if r["budget"] == 128} for path in files]
        pools = {name: [] for name in ("left_only", "right_only", "format", "cutoff")}
        for key in truth:
            left, right = a[key, "numeric"], b[key, "numeric"]
            if left["correct"] != right["correct"]:
                pools["left_only" if left["correct"] else "right_only"].append(key)
            if any(item["correct"] and not table[key, "original"]["correct"]
                   for item, table in ((left, a), (right, b))):
                pools["format"].append(key)
            if any(item["parsed"] is None and item["length"] == 128 for item in (left, right)):
                pools["cutoff"].append(key)
        used = set()
        for cause, pool in pools.items():
            pool = sorted(set(pool) - used)
            for key in rng.sample(pool, min(2, len(pool))):
                used.add(key)
                pair = [("equation", a[key, "numeric"]), ("answer", b[key, "numeric"])]
                rng.shuffle(pair)
                student = [dict(text=row["text"], parsed=row["parsed"], length=row["length"]) for _, row in pair]
                case_id = f"{seed}-{len(cases)+1:02d}"
                cases.append(dict(case=case_id, id=key, gold=truth[key], responses=student))
                keys.append(dict(case=case_id, reason=cause, methods=[method for method, _ in pair]))
    args.directory.mkdir(parents=True, exist_ok=False)
    write_rows(args.directory / "blind-cases.jsonl", cases)
    write_rows(args.directory / "case-key.jsonl", keys)


if __name__ == "__main__":
    main()
