"""Create a locked SVAMP arithmetic transfer sample for a 128-token test."""
import argparse
import hashlib
import json
import unicodedata
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen


REVISION = "78e727689e1c1bebfc4be39c446898e8e10b0518"
SOURCE = f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{REVISION}/SVAMP.json"
SOURCE_SHA256 = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
SAMPLE_SIZE = 256


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(problem):
    return " ".join(unicodedata.normalize("NFKC", problem).lower().split())


def write_lines(path, rows):
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gsm8k_train", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train = {normalize(json.loads(line)["problem"])
             for line in args.gsm8k_train.read_text(encoding="utf-8").splitlines()}
    with urlopen(SOURCE, timeout=30) as stream:
        payload = stream.read()
    if digest(payload) != SOURCE_SHA256:
        raise ValueError("Unexpected source dataset hash")
    source = json.loads(payload, parse_float=Decimal)
    examples = {}
    overlap = 0
    for row in source:
        question = f"{row['Body'].strip()} {row['Question'].strip()}"
        key = normalize(question)
        if key in train:
            overlap += 1
            continue
        identity = str(row["ID"])
        index = digest(f"opsd-svamp-128|{identity}".encode())
        result = dict(id=index, source_id=identity, problem=question, answer=format(row["Answer"], "f"),
                      type=row["Type"])
        previous = examples.setdefault(key, result)
        if Decimal(previous["answer"]) != Decimal(result["answer"]):
            raise ValueError(f"Conflicting answers for {identity}")
    selected = sorted(examples.values(), key=lambda row: row["id"])[:SAMPLE_SIZE]
    if len(selected) != SAMPLE_SIZE:
        raise ValueError("Insufficient unique questions")
    args.output.mkdir(parents=True, exist_ok=False)
    write_lines(args.output / "inputs.jsonl", [{"id": row["id"], "problem": row["problem"]} for row in selected])
    write_lines(args.output / "labels.jsonl", selected)
    manifest = dict(source=SOURCE, revision=REVISION, source_sha256=SOURCE_SHA256, source_count=len(source),
                    selection="first sorted SHA-256 IDs after normalized GSM8K-train exclusion and deduplication",
                    gsm8k_train_sha256=digest(args.gsm8k_train.read_bytes()), removed_overlap=overlap,
                    sample_size=len(selected), inference_budget=128,
                    primary_comparison="answer-only OPSD versus equation-chain SFT under numeric scoring",
                    seeds=[17, 29],
                    sha256={file.name: digest(file.read_bytes()) for file in sorted(args.output.glob("*.jsonl"))})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
