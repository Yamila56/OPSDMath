"""Audit the cached notes after generation without altering training samples."""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from opsdmath.data.gsm8k import number, read_rows
from opsdmath.training.prompts import BRIEF_CONTEXTS


NUMBERS = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,/]\d+)*(?!\w)")
ANSWER_LINE = re.compile(r"(?:^|\n)\s*(?:final answer|answer|result)\s*[:=]\s*(.+)", re.I)


def values(text):
    return {value for match in NUMBERS.findall(text) if (value := number(match)) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("notes", "references", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    references = {r["id"]: number(r["answer"]) for r in read_rows(args.references)}
    rows = read_rows(args.notes)
    if {r["id"] for r in rows} != references.keys() or len(rows) != len(references):
        raise ValueError("Training questions and cached notes differ")
    checks = {mode: Counter() for mode in BRIEF_CONTEXTS}
    flagged = []
    for row in rows:
        in_question = values(row["problem"])
        for mode in BRIEF_CONTEXTS:
            note = row[mode]
            in_note = values(note)
            markers = all(re.search(rf"^\s*{heading}:", note, re.M | re.I)
                          for heading in ("Target", "Notes", "Check"))
            explicit = ANSWER_LINE.search(note)
            checks[mode].update(dict(
                n=1, all_headings=markers, any_new_number=bool(in_note-in_question),
                reference_number_outside_question=references[row["id"]] in in_note-in_question,
                explicit_reference_answer=bool(explicit and references[row["id"]] in values(explicit[1])),
            ))
            if not markers or in_note-in_question or explicit:
                flagged.append(dict(id=row["id"], mode=mode, question=row["problem"], note=note,
                                    gold=str(references[row["id"]]), new_numbers=sorted(map(str, in_note-in_question))))
    rng = random.Random(13592)
    sample = rng.sample(rows, k=min(len(rows), 24))
    args.output.write_text(json.dumps(dict(
        automatic_flags=checks, flagged=flagged, paired_review_sample=sample,
    ), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
