"""Recount the archived EOS direction audit from reference-ending positions."""

import json

from opsdmath.io import read_rows


def verify_eos_audit(folder):
    expected = json.loads((folder / "eos-paired.json").read_text(encoding="utf-8"))
    measured = {}
    for path in (folder / "positions.jsonl", folder / "checkpoints/positions.jsonl"):
        for row in read_rows(path):
            if row["kind"] != "eos":
                continue
            group = measured.setdefault(row["scale"], {})
            for objective, value in row["eos_logprob_direction"].items():
                counts = group.setdefault(objective, {"raises": 0, "lowers": 0})
                counts["raises"] += value > 0
                counts["lowers"] += value < 0
    if measured != {key: value for key, value in expected.items() if "clipped" in value}:
        raise ValueError("Replayed EOS direction counts differ from the published audit")
    return {checkpoint: row["clipped"]["lowers"] for checkpoint, row in measured.items()}
