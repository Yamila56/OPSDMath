"""Record teacher context transformations and prompt lengths before training."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from gsm8k import number, read_rows, write_rows
from run_math import EQUATIONS, prompt, teacher_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = []
    for row in read_rows(args.data):
        equations = EQUATIONS.findall(row["solution"])
        last_value = number(equations[-1].split("=")[-1]) if equations else None
        item = dict(id=row["id"], operation_count=len(equations),
                    last_number_matches_answer=last_value == number(row["answer"]) if last_value is not None else None)
        gold = f"Final answer: {row['answer']}"
        student_ids = prompt(tokenizer, row)[0].tolist()
        for mode in ("full", "equations", "answer"):
            expected = teacher_context(row, mode)
            teacher_ids = prompt(tokenizer, row, privileged=True, context=mode)[0]
            if not expected.endswith(gold) or "Reference Solution Begin" in tokenizer.decode(student_ids):
                raise ValueError(f"Prompt audit failed for {row['id']}/{mode}")
            if prompt(tokenizer, row, context=mode)[0].tolist() != student_ids:
                raise ValueError(f"Teacher mode changed student prompt for {row['id']}")
            item[mode] = dict(text=expected, prompt_tokens=len(teacher_ids))
        report.append(item)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_rows(args.output, report)


if __name__ == "__main__":
    main()
