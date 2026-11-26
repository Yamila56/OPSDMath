"""Student instructions and privileged teacher context."""

import re


INSTRUCTION = "Solve briefly. End with Final answer: <number>."


EQUATIONS = re.compile(r"<<([^<>]+)>>")


BRIEF_CONTEXTS = ("general_brief", "constraint_brief")


def teacher_context(row, mode):
    if mode in BRIEF_CONTEXTS:
        return row[mode]
    if mode == "full":
        return row["solution"]
    final = f"Final answer: {row['answer']}"
    if mode == "answer":
        return final
    return "\n".join([*EQUATIONS.findall(row["solution"]), final])


def prompt(tokenizer, row, privileged=False, context="full", output_budget=None, *,
           key_steps=None, thinking=False):
    problem = row["problem"] if "problem" in row else row["question"]
    message = f"Problem: {problem}\n\n"
    if privileged and context in BRIEF_CONTEXTS:
        message += f"Problem notes:\n{teacher_context(row, context)}\n\n"
    elif privileged and context != "none":
        message += (
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{teacher_context(row, context)}\n=== Reference Solution End ===\n\n"
            "After reading the reference, derive the answer in your own words.\n"
        )
    if key_steps is None:
        key_steps = output_budget is not None
    guidance = []
    if output_budget is not None:
        guidance.append(f"The complete response must fit within {output_budget} tokens.")
    if key_steps:
        guidance.append("Prioritize the key steps.")
    if guidance:
        message += " ".join(guidance) + "\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message + INSTRUCTION}],
        tokenize=True, add_generation_prompt=True, enable_thinking=thinking, return_tensors="pt",
    )
