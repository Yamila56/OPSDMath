"""Compare matched clipped and full-KL pilots on frozen-base budget cohorts."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from gsm8k import read_rows
from retention_study import cohort
from summarize_scale_budget_crossed import interval

GROUPS = ("base", "clipped", "full")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root):
    evaluation = root/"evaluation"
    strata = cohort(evaluation/"base")
    modes = {}
    for mode in GROUPS:
        folder = evaluation/mode
        tokens = {r["id"]:r for r in read_rows(folder/"tokens.jsonl")}
        short, long = (read(folder/f"score-{cap}.json") for cap in (128,512))
        matrices = {f"{rule}_{cap}:{key}":{r["id"]:int(r[key]) for r in value[rule]}
                    for cap,value in ((128,short),(512,long)) for rule in ("numeric","strict")
                    for key in ("correct","parsed")}
        matrices["eos_by_128"] = {i:int(len(r["token_ids"])<=128 and not r["hit_cap"]) for i,r in tokens.items()}
        matrices["short_tokens"] = {i:min(len(r["token_ids"]),128) for i,r in tokens.items()}
        matrices["long_tokens"] = {i:len(r["token_ids"]) for i,r in tokens.items()}
        if any(matrix.keys()!=strata.keys() for matrix in matrices.values()):
            raise ValueError("Evaluation groups do not match frozen cohort IDs")
        modes[mode] = matrices
    cohorts = {"all":list(strata)}
    cohorts.update({s:[key for key,val in strata.items() if val==s] for s in sorted(set(strata.values()))})
    results = {}
    for name,whitelist in cohorts.items():
        results[name] = {"n":len(whitelist),"modes":{},"full_vs_clipped":{}}
        for mode,measurements in modes.items():
            results[name]["modes"][mode] = {
                metric:mean(values[i] for i in whitelist) if metric.endswith("_tokens") else
                sum(values[i] for i in whitelist) for metric,values in measurements.items()
            }
        for key in modes["full"]:
            if key.endswith("_tokens"):
                continue
            values = np.array([modes["full"][key][i]-modes["clipped"][key][i] for i in whitelist])
            results[name]["full_vs_clipped"][key] = dict(**interval(values), improved=int((values>0).sum()),
                                                         regressed=int((values<0).sum()))
    training = {}
    for mode in ("clipped","full"):
        rows = read_rows(root/mode/"events.jsonl")
        updates = [r for r in rows if r["event"]=="update"]
        micro = [r for r in rows if r["event"]=="microstep"]
        if len(updates)!=64 or len(micro)!=256:
            raise ValueError(f"Unexpected training duration for {mode}")
        training[mode] = dict(updates=len(updates), samples=len(micro), tokens=sum(r["tokens"] for r in micro),
                              mean_seconds_per_update=rows[-1]["seconds"]/len(updates),
                              at_cap=sum(r["hit_cap"] for r in micro), seconds=rows[-1]["seconds"])
    return dict(results=results,training=training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.run),indent=2)+"\n",encoding="utf-8")
