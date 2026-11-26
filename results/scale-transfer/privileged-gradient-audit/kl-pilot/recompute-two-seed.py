"""Recompute both-seed and transfer pairings from the archived pilot predictions."""
import json
import sys
from pathlib import Path

import numpy as np

from retention_study import cohort
from summarize_scale_budget_crossed import interval

root = Path(sys.argv[1])
replay = {17:root/"seed17"/"evaluation",29:root/"evaluation"}
data = cohort(replay[17]/"base")
assert data == cohort(replay[29]/"base")
ids = list(data)
read = lambda folder,cap,key="numeric": {
    row["id"]:row["correct"] for row in json.loads((folder/f"score-{cap}.json").read_text())[key]
}
stack = {}
for method in ("base","clipped","full","equations"):
    for cap in (128,512):
        for scorer in ("numeric","strict"):
            rows = []
            for seed in (17,29):
                folder = replay[seed]/method
                if method == "equations":
                    folder = root/"seed17"/"equation-controls"/f"equations-{seed}"
                predictions = read(folder,cap,scorer)
                assert predictions.keys()==data.keys()
                rows.append([predictions[i] for i in ids])
            stack[method,cap,scorer] = np.array(rows,dtype=np.int8)
summary = {}
for cap,scorer in ((128,"numeric"),(128,"strict"),(512,"numeric")):
    for key in ("all","early","late","wrong_both"):
        mask = np.array([key=="all" or data[i]==key for i in ids])
        counts = {m:stack[m,cap,scorer][:,mask].sum(1).tolist()
                  for m in ("base","clipped","full","equations")}
        versus = {}
        for comp in ("base","clipped","equations"):
            comparison = stack["full",cap,scorer][:,mask]-stack[comp,cap,scorer][:,mask]
            versus[comp] = dict(pooled=interval(comparison.mean(0)),by_seed=[interval(x) for x in comparison])
        summary[f"{cap}/{scorer}/{key}"] = dict(counts=counts,n=int(mask.sum()),full_vs=versus)
folder=root/"seed17"/"transfer-evaluation"
transfer={}
for cap in (128,512):
    base = read(folder/"base",cap)
    keys=list(base)
    transfer[str(cap)] = dict(base=sum(base.values()), n=len(base), models={})
    for seed in (17,29):
        learner=read(folder/f"full-{seed}",cap)
        assert learner.keys()==base.keys()
        diff=np.array([int(learner[k])-int(base[k]) for k in keys])
        transfer[str(cap)]["models"][str(seed)] = dict(correct=sum(learner.values()),vs_base=interval(diff))
summary["SVAMP"]=transfer
(root/"paired-two-seed.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
