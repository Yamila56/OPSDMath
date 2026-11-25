"""Audit explicit answer lines missed by the primary boxed-only MATH-500 scorer."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer

from gsm8k import FINAL_MARKER, read_rows, write_rows
from math500_budget import parse_box


def explicit_final(text, capped):
    markers = list(FINAL_MARKER.finditer(text))
    if not markers:
        return None
    remainder = text[markers[-1].end():]
    line, newline, _ = remainder.partition("\n")
    if not newline and capped:
        return None
    return line.strip().rstrip("*").strip().removesuffix(".").strip().strip("$") or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run/"complete.json").read_text(encoding="utf-8"))
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Answer audit uses different gold data")
    gold = {row["unique_id"]: parse_box(r"\boxed{" + row["answer"] + "}") for row in read_rows(args.data)}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    predictions = {row["id"]: row for row in read_rows(args.run/"tokens.jsonl")}
    result = {}
    for budget in (128, 256, 512):
        baseline = read_rows(args.run/f"scored-{budget}.jsonl")
        if len(baseline) != len(gold) or {r["id"] for r in baseline} != gold.keys() or predictions.keys() != gold.keys():
            raise ValueError("Answer audit must use the same rows as the original scorer")
        records = []
        for row in baseline:
            key = row["id"]
            prediction = predictions[key]
            tokens = prediction["token_ids"]
            recovered = False
            final = None
            status = ("no_closed_box" if not row["closed_box"] else
                      "unparsed_box" if not row["parsed"] else
                      "correct_box" if row["correct"] else "incorrect_box")
            if status != "correct_box":
                capped = len(tokens) > budget or (len(tokens) == budget and prediction["hit_cap"])
                final = explicit_final(tokenizer.decode(tokens[:budget], skip_special_tokens=True), capped)
                if final:
                    parsed = parse("$"+final+"$", extraction_config=[LatexExtractionConfig()],
                                   fallback_mode="no_fallback", parsing_timeout=2)
                    recovered = bool(parsed and verify(gold[key], parsed, timeout_seconds=2))
            records.append(dict(id=key, box_status=status, explicit_final=final is not None,
                                recovered=recovered, correct=row["correct"] or recovered,
                                correct_with_parse_fallback=row["correct"] or
                                (recovered and status != "incorrect_box")))
        write_rows(args.run/f"delivery-{budget}.jsonl", records)
        result[str(budget)] = dict(
            explicit_without_box=sum(r["explicit_final"] and r["box_status"] == "no_closed_box" for r in records),
            explicit_by_box_status=dict(Counter(r["box_status"] for r in records if r["explicit_final"])),
            recovered_by_box_status=dict(Counter(r["box_status"] for r in records if r["recovered"])),
            recovered=sum(r["recovered"] for r in records),
            correct_with_parse_fallback=sum(r["correct_with_parse_fallback"] for r in records),
            correct_with_either_candidate=sum(r["correct"] for r in records),
        )
    (args.run/"delivery.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
