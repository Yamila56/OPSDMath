"""Construct fresh condition triples with explicit, independently computed gold answers."""
import argparse
import random
from pathlib import Path

from gsm8k import write_rows


def scenario(index, a, b, c):
    cases = [
        (f"Lin purchases {b} notebooks at an equal price and pays {c*b} dollars in total.",
         f"The receipt shows that the payment includes a {b}-dollar service fee.",
         f"Lin also receives {b} blank shopping coupons with no cash value.",
         "What is the price of one notebook in dollars before any service fee?",
         c, c-1),
        (f"A workshop has {a} people. Each person completes {b} pieces per day for {c} days.",
         "On the last day, two people are absent and complete no pieces.",
         "On the last day, two people wear blue aprons.",
         "How many pieces are completed?", a*b*c, a*b*c-2*b),
        (f"A reading contest awards {a} points for a novel and {b} points for a poem. Jo reads {c} of each.",
         "The points for each poem are doubled.",
         "Jo rereads each poem, with no extra points for rereading.",
         "How many points does Jo earn?", c*(a+b), c*(a+2*b)),
        (f"A rectangle is {a} meters long and {b} meters wide. Its entire outside edge needs fencing.",
         f"An opening of {c} meters will be left unfenced along one long side.",
         f"There are {c} trees inside the rectangle.",
         "How many meters of fence are needed?", 2*(a+b), 2*(a+b)-c),
        (f"Each of {a} tables has exactly {b} seated guests. Every guest gets one drink.",
         f"At each table, {c} guests ask for a second drink.",
         f"At each table, {c} guests put their coats on the back of their chairs.",
         "How many drinks are served?", a*b, a*(b+c)),
        (f"Two machines make {a} and {b} parts per hour, respectively. Each runs for {c} hours.",
         "The slower machine runs for one additional hour.",
         "The slower machine is one year older.",
         "How many parts are made in total?", c*(a+b), c*(a+b)+min(a,b)),
        (f"A cashier uses bills worth {a} dollars and coins worth {b} dollars to assemble two identical gift envelopes. "
         f"Each envelope contains {c} of each item.",
         f"One coin in each envelope is replaced by a bill worth {a} dollars.",
         "The name of a different recipient is written on each envelope.",
         "What is the combined value of both envelopes in dollars?", 2*c*(a+b), 2*c*(a+b)+2*(a-b)),
        (f"A staircase has {a} steps of height {b} centimeters each.",
         f"The bottom step is rebuilt to be {c} centimeters higher.",
         f"The bottom step is labeled with the number {c}.",
         "What is the vertical rise from before the first step to after the last step, in centimeters?",
         a*b, a*b+c),
    ]
    return cases[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-family", type=int, default=12)
    args = parser.parse_args()
    rng = random.Random(221973)
    inputs, labels = [], []
    for family in range(8):
        available = [(a, b, c) for a in range(12, 24) for b in range(2, 10)
                     for c in range(2, min(b, 5)+1)]
        rng.shuffle(available)
        distinct = {}
        for values in available:
            result = scenario(family, *values)
            distinct.setdefault(result[0], result)
        if args.per_family > len(distinct):
            raise ValueError(f"Too many requested examples for family {family}")
        for index, (story, changed, inert, question, original, modified) in enumerate(
            list(distinct.values())[:args.per_family]
        ):
            if modified == original:
                raise ValueError(f"Nonintervention in family {family}")
            pair = f"confirm-{family:02}-{index:03}"
            for variant, clause, label in (("base", "", original), ("changed", changed, modified),
                                           ("inert", inert, original)):
                key = pair + "-" + variant
                inputs.append(dict(id=key, question=" ".join(x for x in (story, clause, question) if x)))
                labels.append(dict(id=key, variant=variant, pair=pair, family=family, answer=str(label),
                                   baseline=str(original)))
    if len({r["question"] for r in inputs}) != len(inputs):
        raise ValueError("Duplicate question text")
    rng.shuffle(inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output/"inputs.jsonl", inputs)
    write_rows(args.output/"labels.jsonl", labels)


if __name__ == "__main__":
    main()
