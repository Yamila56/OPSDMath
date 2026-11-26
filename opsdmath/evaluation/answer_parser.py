"""Numeric and strict final-answer parsers."""

import re
import unicodedata
from fractions import Fraction


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


INLINE_FINAL = re.compile(r"\bFinal answer\**[ \t]*:[ \t]*\**[ \t]*([^\r\n]+)", re.IGNORECASE)


UNIT_ENDING = re.compile(
    r"\s+(?:seconds?|minutes?|hours?|days?|weeks?|years?|inches?|feet|foot|yards?|meters?|miles?|mph|"
    r"cents?|dollars?|pounds?|grams?|kilograms?|liters?|gallons?)\.?\**$", re.IGNORECASE,
)


def audit_final_answer(response):
    """Retrospective format check on explicit final lines with inline markers or plain-language units."""
    primary = parse_answer(response)
    answers = INLINE_FINAL.findall(response)
    if primary is not None or len(answers) != 1:
        return primary
    return decorated_number(UNIT_ENDING.sub("", answers[0].strip()))
