"""Evaluate the same generated trajectory at fixed token limits."""

from statistics import mean

from opsdmath.io import read_rows
from opsdmath.evaluation.answer_parser import audit_final_answer, parse_answer, parse_strict_answer

PARSERS = {"numeric": audit_final_answer, "original": parse_answer, "strict": parse_strict_answer}


def validate_rows(source, labels, expected):
    questions, answers = read_rows(source), read_rows(labels)
    ids = {r["id"] for r in questions}
    if len(ids) != expected or len(questions) != expected or len(answers) != expected:
        raise ValueError(f"Expected {expected} unique evaluation questions in {source}")
    if {r["id"] for r in answers} != ids or any({"answer", "solution"} & r.keys() for r in questions):
        raise ValueError(f"Labels must match the question-only set {source}")
    return questions, answers


def score(responses, truth, tokenizer, cap):
    token_lists = [row["token_ids"][:cap] for row in responses]
    texts = tokenizer.batch_decode(token_lists, skip_special_tokens=True)
    scores, counts = {}, {}
    for key, parse in PARSERS.items():
        status = []
        for row, text in zip(responses, texts, strict=True):
            answer = parse(text)
            status.append(dict(id=row["id"], correct=answer == truth[row["id"]], parsed=answer is not None))
        scores[key] = status
        counts[key] = dict(n=len(status), correct=sum(r["correct"] for r in status), parsed=sum(r["parsed"] for r in status),
                           mean_generated_tokens=mean(map(len, token_lists)),
                           hit_cap=sum(len(row["token_ids"]) > cap or
                                       (len(row["token_ids"]) == cap and row["hit_cap"]) for row in responses))
    return counts, scores
