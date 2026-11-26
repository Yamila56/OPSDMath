"""Compare shared greedy draft prefixes and account for recovered numeric answers."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from opsdmath.data.gsm8k import number, read_rows
from score_structured_answer import audit

# Match explicit conclusions rather than every occurrence of the gold number.
CONCLUSION = re.compile(
    r"(?:final answer\s*:\s*|\\boxed\{)\s*(?:\$+\s*)?(?:<)?([+-]?\d+(?:[./]\d+)?)"
    r"(?=\s*(?:[}>$]|$|\n|[.,!?](?!\d)))",
    re.IGNORECASE,
)


def conclusion_position(tokenizer, tokens, gold):
    text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for match in CONCLUSION.finditer(text):
        if number(match[1]) != gold:
            continue
        # Positions are one-based and include the entire explicit numeric span.
        for position in range(1, len(tokens)+1):
            prefix = tokenizer.decode(tokens[:position], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if len(prefix) >= match.end() and prefix.startswith(text[:match.end()]):
                return position
    return None


def load_predictions(path, expected):
    rows = read_rows(path)
    indexed = {r["id"]: r for r in rows}
    if len(rows) != len(expected) or indexed.keys() != expected:
        raise ValueError(f"Incorrect evaluation IDs in {path}")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "labels", "runs", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("methods", nargs="+")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    labels = {r["id"]: r for r in read_rows(args.labels)}
    report = {}
    for method in args.methods:
        low, high = (load_predictions(args.runs/f"test-{budget}"/method/"predictions.jsonl", labels.keys())
                     for budget in (128, 512))
        stats, scope, recoveries = Counter(), Counter(), []
        for key, label in labels.items():
            a, b = low[key], high[key]
            gold = number(label["answer"])
            low_value = audit(a, budget=128, draft_cap=104)
            high_value = audit(b, budget=512, draft_cap=488)
            low_tokens, high_tokens = a["draft_token_ids"], b["draft_token_ids"]
            prefix_matches = low_tokens == high_tokens[:104]
            scoped = label["variant"] == "relation"
            for counts in (stats, scope) if scoped else (stats,):
                counts.update(dict(n=1, prefix_match=prefix_matches,
                                   low_correct=low_value == gold, high_correct=high_value == gold,
                                   low_complete=low_value is not None, high_complete=high_value is not None,
                                   low_cap=a["draft_hit_cap"], high_cap=b["draft_hit_cap"],
                                   losses=low_value == gold and high_value != gold))
            if low_value == gold or high_value != gold:
                continue
            first = conclusion_position(tokenizer, high_tokens, gold)
            recovery = "incomplete_short_field" if low_value is None else "wrong_short_field"
            position = "no_explicit_draft_conclusion" if first is None else (
                "by_short_draft_cap" if first <= 104 else "after_short_draft_cap")
            for counts in (stats, scope) if scoped else (stats,):
                counts.update([recovery, position])
            recoveries.append(dict(id=key, variant=label["variant"], short_field=recovery,
                                   short_draft_at_cap=a["draft_hit_cap"], prefix_match=prefix_matches,
                                   long_draft_correct_conclusion_position=first,
                                   long_final_eos=b["final_eos"], long_generated_tokens=b["generated_tokens"]))
        report[method] = dict(all=stats, relation=scope, recoveries=recoveries)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
