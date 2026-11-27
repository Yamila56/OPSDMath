"""Render the OPSDMath framework diagram and data figures from checked CSVs."""
import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

PAPER = "#ffffff"
INK = "#172431"
MUTED = "#4d5b69"
LINE = "#d8e1e9"
NAVY = "#3c5488"
SKY = "#e5f1fd"
PINK = "#fce6ef"
MINT = "#e0f5e9"
LILAC = "#eeeafa"
SAND = "#fff2d9"
COLORS = {"base":"#68758a", "sft":"#5486b8", "online":"#148067",
          "equation":"#9f4773", "answer":"#89611f", "fixed":"#6853a2"}
FILLS = {"base":"#f1f4f8", "sft":SKY, "online":MINT, "equation":PINK, "answer":SAND, "fixed":LILAC}
NAMES = {"base":"Base", "sft":"SFT", "online":"Full online",
         "equation":"Equations + answer", "answer":"Answer only", "fixed":"Full fixed rollout"}
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "figure.facecolor": PAPER})


def label(fig, x, y, name, size=9.5, color=INK, bold=False, align="left"):
    fig.text(x, y, name, color=color, size=size, weight="semibold" if bold else "normal",
             ha=align, va="center")


def title(fig, name, detail):
    fig.text(.034,.954,name,size=20,weight="bold",fontfamily=["Cambria","STIXGeneral","DejaVu Serif"],
             color=INK,va="center")
    label(fig,.034,.905,detail,size=8.9,color=MUTED)


def box(fig, x, y, w, h, color, edge=LINE):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=color,
                             edgecolor=edge,lw=.7,zorder=0))


def arrow(fig, start, end, dotted=False):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>",
           mutation_scale=11, lw=.9, color=MUTED, linestyle=":" if dotted else "-", zorder=3))


def save(fig, path):
    fig.savefig(path,dpi=224,facecolor=PAPER)
    plt.close(fig)


def workflow(path):
    fig = plt.figure(figsize=(10.5,5.55))
    fig.text(.034,.95,"What should the teacher see?",fontfamily="Segoe UI",size=19,
             weight="medium",color=INK,va="center")
    label(fig,.034,.898,"OPSDMath · Qwen3-1.7B · 256 training problems · two training seeds",size=9.9,color=MUTED)
    lanes = ((.034,.264,SKY,"A   Prepare the inputs"),
             (.314,.371,PINK,"B   Distill on student trajectories"),
             (.701,.265,MINT,"C   Evaluate at fixed token budgets"))
    for x,w,fill,heading in lanes:
        box(fig,x,.146,w,.685,PAPER)
        box(fig,x,.753,w,.078,fill)
        label(fig,x+.013,.790,heading,size=10.6,color=NAVY)
    box(fig,.049,.614,.233,.123,SKY)
    label(fig,.060,.706,"GSM8K question",bold=True)
    label(fig,.060,.655,"Example: starting balance $126.\nGroceries $60, gas half as much.\nThen a $45 return. New balance?",size=7.9,color=MUTED)
    label(fig,.052,.560,"Teacher receives one extra context",size=8.55,bold=True)
    for y,fill,name,note in ((.430,MINT,"Full solution","Original rationale and answer"),
                            (.314,PINK,"Equations + answer","60/2=30 … 216−45=171"),
                            (.198,SAND,"Answer only","Final answer: 171")):
        box(fig,.049,y,.233,.091,fill)
        label(fig,.060,y+.059,name,bold=True,size=8.8)
        label(fig,.060,y+.026,note,size=8.25,color=MUTED)
    arrow(fig,(.298,.776),(.314,.776))
    for x,name,note,fill in ((.330,"Student","Question only\nTrainable LoRA",SKY),
                             (.508,"Teacher","Question and chosen context\nFrozen shared base",PAPER)):
        box(fig,x,.610,.161,.125,fill)
        label(fig,x+.0805,.700,name,bold=True,align="center")
        label(fig,x+.0805,.647,note,size=8.05,color=MUTED,align="center")
    arrow(fig,(.410,.610),(.410,.554))
    box(fig,.330,.479,.339,.074,PAPER)
    label(fig,.4995,.517,"Sample a rollout from the current student",size=8.7,bold=True,align="center")
    fig.add_artist(Line2D([.669,.677,.677],[.672,.672,.397], transform=fig.transFigure,
                          linestyle=":",color=MUTED,lw=.9))
    arrow(fig,(.677,.397),(.669,.397),dotted=True)
    arrow(fig,(.4995,.477),(.4995,.435))
    box(fig,.330,.351,.339,.084,PAPER)
    label(fig,.4995,.400,"Compare logits along the same rollout",size=8.7,align="center")
    label(fig,.4995,.370,"Clipped forward KL updates student LoRA",size=8.25,color=MUTED,align="center")
    box(fig,.330,.205,.339,.098,LILAC)
    label(fig,.342,.267,"Fixed-rollout control",size=8.9,bold=True)
    label(fig,.342,.232,"Sample once from the initial student",size=8.2,color=MUTED)
    arrow(fig,(.685,.776),(.701,.776))
    for y,name,note in ((.635,"Decode","Question-only, greedy generation"),
                        (.514,"Score the four prefixes","64   /   128   /   256   /   512 tokens"),
                        (.393,"Measure answer delivery","Numeric accuracy, parse rate, cutoffs")):
        box(fig,.716,y,.235,.093,PAPER)
        label(fig,.727,y+.060,name,bold=True,size=8.75)
        label(fig,.727,y+.026,note,size=8.0,color=MUTED)
    label(fig,.716,.306,"Base and SFT baselines",size=8.75,bold=True)
    label(fig,.716,.247,"256 historical diagnostic questions\n512 held-out confirmation questions",
          size=8.1,color=MUTED)
    label(fig,.034,.063,"The teacher context is used only in training. Each new configuration runs 64 updates on one GPU.",
           size=7.8,color=MUTED)
    save(fig,path)


def csvrows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def data_figure(source, path, sample_name):
    raw = [r for r in csvrows(source) if r["parser"] == "numeric"]
    groups = [method for method in NAMES
              if any(r["method"].rstrip("0123456789") == method for r in raw)]
    budgets = (64,128,256,512)
    n = int(raw[0]["n"])
    fig, axes = plt.subplots(1,2,figsize=(10.5,4.7))
    title(fig,"Accuracy and answer delivery under token limits",
          f"{sample_name} · {n} questions per model · curves average seeds when two adapters are available")
    fig.subplots_adjust(top=.79,bottom=.23,left=.094,right=.972,wspace=.245)
    for axis,metric,head in zip(axes,("accuracy","parsed"),("A   Numeric accuracy","B   Parseable answers"),strict=True):
        for method in groups:
            vals = [[float(row[metric])/ (n if metric == "parsed" else 1) * 100
                     for row in raw if row["method"].rstrip("0123456789")==method and int(row["budget"])==budget]
                    for budget in budgets]
            if not all(vals):
                continue
            vals = [mean(items) for items in vals]
            marker = {"base":"x","sft":"o","online":"s","equation":"D","answer":"^","fixed":"v"}[method]
            axis.plot(range(4),vals,color=COLORS[method],marker=marker,ms=6,lw=1.65,
                      mfc=FILLS[method],mew=1.2,label=NAMES[method])
        axis.set(xticks=range(4),xticklabels=budgets,ylim=(-2,103),yticks=[0,25,50,75,100])
        axis.set_title(head,loc="left",size=10.6,weight="bold",color=NAVY)
        axis.set_ylabel("Questions (%)",size=8.6,color=MUTED)
        axis.set_xlabel("Maximum generated tokens",size=8.55,color=MUTED)
        axis.tick_params(axis="both",colors=MUTED,labelsize=8.6,length=0)
        axis.spines[:].set_visible(False)
        axis.grid(axis="y",color=LINE,lw=.75)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.53,.042),ncol=min(len(groups),3),
               frameon=False,fontsize=8.5,labelcolor=MUTED)
    save(fig,path)


def table_128(source,path,sample_name):
    rows=[row for row in csvrows(source) if row["parser"] == "numeric" and row["budget"] == "128"]
    by_name={row["method"]:row for row in rows}
    ordered=[m for m in NAMES if m in by_name or m+"17" in by_name and m+"29" in by_name]
    n=int(rows[0]["n"])
    fig=plt.figure(figsize=(9.2,1.88 + .42*len(ordered)))
    title(fig,"What changes at 128 tokens?",
          f"{sample_name} · {n} questions per seed · numeric scoring with format tolerance")
    cols=(.045,.585,.718,.841,.942)
    headings=("Training condition","Acc. 17","Acc. 29","Parsed 17","Parsed 29")
    for i,(x,head) in enumerate(zip(cols,headings,strict=True)):
        label(fig,x,.784,head,size=9.4,color=NAVY,bold=True,align="left" if i==0 else "right")
    stride=.57/len(ordered)
    for index,group in enumerate(ordered):
        y=.675-index*stride
        row_h=stride*.88
        box(fig,.034,y-row_h/2,.933,row_h,FILLS[group],FILLS[group])
        pair=[by_name.get(f"{group}{seed}") for seed in (17,29)]
        if group=="base":
            pair=[by_name["base"],None]
        label(fig,cols[0],y,"Base (shared)" if group=="base" else NAMES[group],size=9.3,bold=True)
        for x,val in zip(cols[1:], [f"{100*float(r['accuracy']):.1f}%" if r else "—" for r in pair] +
                        [f"{int(r['parsed'])}/{n}" if r else "—" for r in pair],strict=True):
            label(fig,x,y,val,size=9.3,align="right")
    label(fig,.034,.078,"Acc. is correct answers divided by all questions. Parsed counts include wrong numeric answers.",
          size=8.1,color=MUTED)
    save(fig,path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--metrics",type=Path)
    parser.add_argument("--split",choices=("diagnostic","confirmation"),default="diagnostic")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    workflow(args.output/"workflow.png")
    if args.metrics:
        name="Historical diagnostic split" if args.split=="diagnostic" else "Held-out confirmation split"
        data_figure(args.metrics,args.output/f"{args.split}-budget-curves.png",name)
        table_128(args.metrics,args.output/f"{args.split}-128-token-table.png",name)


if __name__=="__main__":
    main()
