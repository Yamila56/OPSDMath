"""Training and prediction run events and device timing."""

import json

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def log(event, output):
    row = json.dumps(event, ensure_ascii=False)
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(row + "\n")
    print(row, flush=True)


def memory(device):
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    } if device.type == "cuda" else {}
