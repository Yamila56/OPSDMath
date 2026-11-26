"""Render the README figures from archived results. Run from any directory."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/figures"
OUT.mkdir(parents=True, exist_ok=True)
teal, blue, gray, gold, purple, red = "#168A83","#3974B8","#64748B","#D8A04B","#8877AE","#C45C58"
ink = "#182230"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"text.color":ink,
    "axes.labelcolor":ink,"xtick.color":gray,"ytick.color":gray,"axes.edgecolor":"#CBD5E1",
    "axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":9,
    "axes.titleweight":"bold","svg.fonttype":"path","svg.hashsalt":"research"})
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def save(fig,name):
    for ext in ("png","svg"):
        fig.savefig(OUT/f"{name}.{ext}",dpi=300,facecolor="white",metadata={"Date":None} if ext=="svg" else None)
    plt.close(fig)
def diagram(height):
    fig=plt.figure(figsize=(8,height))
    ax=fig.add_axes([0,0,1,1],xlim=(0,100),ylim=(0,100))
    ax.axis("off")
    return fig,ax
def label(ax,x,y,s,size=8,color=ink,bold=False,**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight="bold" if bold else "normal",
                   va="center",**kw)
def arrow(ax,x,y,u,v,color=gray,dashed=False):
    ax.annotate("",(u,v),(x,y),arrowprops=dict(arrowstyle="->",color=color,lw=1,
                linestyle="--" if dashed else "-",mutation_scale=9))
def panel(ax,title):
    ax.set_title(title,loc="left",pad=10)

AUDIT=ROOT/"results/scale-transfer/privileged-gradient-audit"
scores=read(AUDIT/"kl-pilot/paired-two-seed.json")
fig,ax=diagram(2.8)
label(ax,3,91,"(a)  One model, two contexts",10,bold=True)
label(ax,52,91,"(b)  Supervision on the student’s prefix",10,bold=True)
ax.plot([49,49],[15,93],color="#E2E8F0")
for y,teacher in [(67,False),(37,True)]:
    label(ax,3,y+7,"Frozen teacher" if teacher else "Online student",8,gray if teacher else blue,True)
    label(ax,3,y-1,"7 + 5?  +  answer 12" if teacher else "7 + 5?",8,gold if teacher else ink)
    arrow(ax,23,y,27,y)
    label(ax,29,y,"LoRA disabled" if teacher else "LoRA enabled",8,gray if teacher else blue)
    for i,word in enumerate(["y₁","y₂","…","yₜ"]):
        ax.add_patch(Rectangle((53+i*4,y-4),3.4,8,color="#F1F4F7"))
        label(ax,54.7+i*4,y,word,8,ha="center")
    arrow(ax,69,y,73,y)
    heights=[4,10,6,2,5,3] if teacher else [7,3,9,4,5,2]
    ax.bar(np.arange(6)*1.45+75,heights,width=1,bottom=y-3,color=gold if teacher else blue)
    label(ax,85,y,"Teacher q" if teacher else "Student p",8)
ax.plot([28,45,45,28],[81,81,26,26],color=gray,lw=.9)
label(ax,28,17,"Shared Qwen3 weights",8,gray,True)
label(ax,53,18,"Example prompt and distributions are schematic",8,gray)
ax.plot([3,97],[11,11],color="#CBD5E1")
label(ax,3,5,"Full KL:  Σ qᵢ log(qᵢ / pᵢ)   or clipped KL",8,teal,True)
arrow(ax,46,5,54,5,blue,True)
label(ax,56,5,"Update student LoRA, then sample again",8,blue)
save(fig,"overview")

fig,axes=plt.subplots(1,3,figsize=(8,2.9),layout="constrained",gridspec_kw={"width_ratios":[1,1,1.35]})
p=np.array([.48,.20,.18,.14]); q=np.array([.18,.52,.20,.10])
c=q*np.log(q/p)
for a,title,values in zip(axes,["(a)  Full forward KL","(b)  Vocabulary-wise clip"],[c,np.minimum(c,.05)]):
    panel(a,title)
    a.axhline(0,color=gray,lw=.7)
    a.bar(range(4),values,color=[gray,teal if a is axes[0] else red,gray,gray])
    a.set(xticks=range(4),xticklabels=list("abcd"),ylim=(-.25,.68),xlabel="Vocabulary item")
axes[0].set_ylabel("Loss contribution")
axes[0].text(.02,.96,r"$c_i=q_i\log(q_i/p_i)$",transform=axes[0].transAxes,va="top")
axes[1].axhline(.05,color=red,ls="--",lw=.8)
axes[1].text(.02,.96,r"$\sum_i\min(c_i,0.05)$",transform=axes[1].transAxes,va="top")
audit=read(AUDIT/"eos-paired.json")
a=axes[2]; panel(a,"(c)  EOS decreases")
keys=["1.7B","1.7B/17","1.7B/29","8B","8B/17","8B/29"]
n=[audit[k]["clipped"]["lowers"] for k in keys]
a.barh(range(6),n,color=[gray]*3+[red]*3)
for i,v in enumerate(n): a.text(v+1,i,str(v),va="center")
a.set(yticks=range(6),yticklabels=[s.replace("/"," · ") for s in keys],xlim=(0,59),xlabel="Reference endings (of 59)")
a.invert_yaxis()
save(fig,"mechanism")

fig,axes=plt.subplots(1,3,figsize=(8,3.35),gridspec_kw={"width_ratios":[1,1,1]},layout="constrained")
methods=[("base","Base",gray),("clipped","Clipped",red),("full","Full KL",teal),("equations","SFT",purple)]
for j,(key,name,color) in enumerate(methods):
    pair=lambda cap,group: np.array(scores[f"{cap}/numeric/{group}"]["counts"][key])
    for i,cap in enumerate([128,512]):
        vals=pair(cap,"all")/192*100
        axes[0].scatter(i+(j-1.5)*.14+np.array([-.02,.02]),vals,s=18,color=color)
    axes[1].scatter(pair(128,"late")/41*100,pair(128,"early")/140*100,color=color,label=name,s=24)
for a,t in zip(axes,["(a)  Budget accuracy","(b)  Behavioral trade-off","(c)  Full KL advantage"]): panel(a,t);a.grid(axis="y",color="#E2E8F0",lw=.5);a.set_axisbelow(True)
axes[0].set(xticks=[0,1],xticklabels=["128","512"],xlim=(-.4,1.4),ylabel="Correct (%)",xlabel="Generated tokens",ylim=(40,100))
axes[1].set(xlabel="Late accelerated (%)",ylabel="Early retained (%)",xlim=(-5,67),ylim=(62,104))
axes[1].legend(loc="lower right",fontsize=6,frameon=False)
a=axes[2]; a.axvline(0,color=gray,lw=.8,ls="--")
for i,key in enumerate(["clipped","base","equations"]):
    e=scores["128/numeric/all"]["full_vs"][key]["pooled"]; l,h=e["interval95_pp"]; m=e["pp"]
    a.plot([l,h],[i,i],color=teal);a.scatter(m,i,color=teal,s=22)
    a.text(m,i-.18,f"+{m:.2f}",ha="center",fontsize=7)
a.set(yticks=range(3),yticklabels=["Clipped","Base","SFT"],xlabel="Difference (pp), 95% CI",ylim=(2.55,-.55))
save(fig,"main_results")
