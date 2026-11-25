"""Generate new condition-pair templates for the inference budget hint comparison."""
import argparse
import random
from pathlib import Path

from gsm8k import write_rows


def scenario(family, a, b, c):
    cases = [
        (f"A bus starts a route with {a} passengers and picks up {b} more at a stop.",
         f"Before leaving the stop, {c} passengers get off.",
         "Before leaving the stop, the driver checks a mirror.",
         "How many passengers are on board when the bus leaves the stop?", a+b, a+b-c),
        (f"A water tank contains {a} liters. A pump adds {b} liters per minute for {c} minutes.",
         "During pumping a drain removes 1 liter per minute.",
         "The pump was serviced one month ago.",
         "How many liters are in the tank when pumping ends?", a+b*c, a+(b-1)*c),
        (f"A library has {a} boxes with {b} books in each, as well as {c} loose books.",
         "The library gives away all books in one box and keeps the loose books.",
         "The library puts a new label on one box.",
         "How many books does the library have at the end?", a*b+c, (a-1)*b+c),
        (f"A school allocates {a*b} dollars equally to {b} clubs.",
         f"An additional donation of {c*b} dollars will be divided equally across the same clubs.",
         f"The clubs meet in {c} different rooms.",
         "How many dollars is each club allocated, including any donation?", a, a+c),
        (f"Mina reads {a} pages on Monday and twice as many pages on Tuesday.",
         f"She also reads {b} pages on Wednesday.",
         f"Her book has {b} illustrations.",
         "How many pages did she read on the days described?", 3*a, 3*a+b),
        (f"A hiker crosses {a} segments, each {b} kilometers long.",
         f"One segment is closed, so the hiker replaces it with a {b+c}-kilometer detour.",
         f"One segment has a trail marker with the number {b+c}.",
         "How many kilometers does the hiker travel?", a*b, a*b+c),
        (f"An assembly station has {a} parts and adds {b} per hour. There are no removals in the first hour.",
         f"In the first hour alone, a late delivery adds {c} additional parts.",
         f"The station operator labels {c} of the existing parts in the first hour.",
         "How many parts are at the station after the first hour?", a+b, a+b+c),
        (f"A cinema sells {a} full-price tickets for {b} dollars each.",
         f"Exactly {c} tickets are refunded in full before the show.",
         f"Exactly {c} customers sit in the front row.",
         "How many dollars does the cinema keep from these ticket sales?", a*b, (a-c)*b),
    ]
    return cases[family]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = random.Random(94574)
    pool = [(a, b, c) for a in range(12, 25) for b in range(5, 11) for c in range(2, 5)]
    inputs, labels = [], []
    for family in range(8):
        stories = set()
        for values in rng.sample(pool, len(pool)):
            story, changed, inert, question, original, modified = scenario(family, *values)
            if story in stories:
                continue
            stories.add(story)
            pair = f"budget-prompt-{family:02}-{len(stories):03}"
            if modified == original:
                raise ValueError("The effective condition must change the answer")
            for variant, addition, answer in (("base", "", original), ("changed", changed, modified),
                                               ("inert", inert, original)):
                key = f"{pair}-{variant}"
                text = " ".join(piece for piece in (story, addition, question) if piece)
                inputs.append(dict(id=key, question=text))
                labels.append(dict(id=key, pair=pair, variant=variant, family=family,
                                   baseline=str(original), answer=str(answer)))
            if len(stories) == 12:
                break
    if len(inputs) != 288 or len({r["question"] for r in inputs}) != len(inputs):
        raise RuntimeError("Expected eight complete families of distinct condition triples")
    rng.shuffle(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output/"inputs.jsonl", inputs)
    write_rows(args.output/"labels.jsonl", labels)


if __name__ == "__main__":
    main()
