import sys; sys.path.insert(0,'.')
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from model import *
from abstract_sim import mc_uniform, ci
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
                     'xtick.major.width':.6,'ytick.major.width':.6,'savefig.dpi':600,'figure.dpi':150})
BLUE,ORG,VIO,GRN,INK,MUT='#2a78d6','#eb6834','#6250d6','#008300','#1c1c1c','#6b6a66'
W=4.7

# ---------------- Fig 1: protocol schematic
fig,ax=plt.subplots(figsize=(W,2.0)); ax.set_xlim(0,100); ax.set_ylim(-1,46); ax.axis('off')
segs=[(3,25),(33,55),(63,85)]; vx=[]
for i,(a,b) in enumerate(segs):
    nstep=4; w=(b-a)/nstep
    for j in range(nstep):
        ax.add_patch(Rectangle((a+j*w,25),w-.7,7,fc='#dce8f8',ec=BLUE,lw=.6))
        ax.text(a+j*w+(w-.7)/2,28.5,f's{i*nstep+j+1}',ha='center',va='center',fontsize=6.5,color=INK)
    ax.add_patch(FancyBboxPatch((b+.8,24.2),5.4,8.6,boxstyle='round,pad=0.15,rounding_size=1.2',fc='#fde6dc',ec=ORG,lw=.7))
    ax.text(b+3.5,28.5,'V',ha='center',va='center',fontsize=8,color=INK,fontweight='bold'); vx.append(b+3.5)
    ax.text(a,36.5,f'segment {i+1}',ha='left',fontsize=6.8,color=MUT)
for x,lab in [(vx[0],'C1'),(vx[1],'C2'),(vx[2],'C3')]:
    ax.plot([x,x],[19.5,24.2],color=INK,lw=.6)
    ax.add_patch(Rectangle((x-2.2,14.5),4.4,5,fc='white',ec=INK,lw=.7)); ax.text(x,17,lab,ha='center',va='center',fontsize=6.3)
ax.add_patch(Rectangle((0.8,14.5),4.4,5,fc='white',ec=INK,lw=.7)); ax.text(3,17,'C0',ha='center',va='center',fontsize=6.3)
ax.plot([3,3],[19.5,25],color=INK,lw=.6)
ax.annotate('',xy=(vx[0]+2.6,17),xytext=(vx[1]-2.6,17),arrowprops=dict(arrowstyle='-|>',color=ORG,lw=.9))
ax.text((vx[0]+vx[1])/2,11.0,'alarm in segment 2: restore C1, run segment 2 again',ha='center',fontsize=6.6,color=ORG)
ax.text(46.75,34.3,'\u00d7',color='#b3261e',fontsize=10,ha='center',va='center',fontweight='bold',zorder=5)
ax.plot([46.75,46.75],[36,40],color='#b3261e',lw=.5,ls=':'); ax.plot([46.75,46.75],[32,33.2],color='#b3261e',lw=.5); ax.text(47.8,41.2,'silent error: nothing visible to the agent',fontsize=6.4,color='#b3261e',ha='left')
ax.text(1,5.0,'cost of one attempt:  k step executions plus one verification (c\u2080 + c\u2081k step units)',fontsize=6.7,color=INK)
ax.text(1,0.0,'verifier recall r (alarm when the segment holds an error), false-alarm rate f (alarm on a clean segment)',fontsize=6.7,color=MUT)
fig.savefig('figs/fig1_protocol.png',bbox_inches='tight',pad_inches=.03); plt.close(fig)

# ---------------- Fig 2: exact cost vs interval, MC overlay
rng=np.random.default_rng(31415)
n=48; c0,c1,r,f=0.5,0.0,0.8,0.05
fig,axs=plt.subplots(1,2,figsize=(W,2.1),gridspec_kw=dict(width_ratios=[1.25,1]))
ax=axs[0]; fig2={}
for p,col,mk in [(0.01,BLUE,'o'),(0.03,ORG,'s'),(0.08,VIO,'^')]:
    ks=np.arange(1,n+1); ex=np.array([uniform_cost(n,k,p,c0,c1,r,f)[0]/n for k in ks])
    ax.plot(ks,ex,color=col,lw=1.1,label=f'p = {p}')
    kmc=[1,2,3,4,6,8,12,16,24,48]; mc=[];mci=[]
    for k in kmc:
        cst,_=mc_uniform(n,k,p,c0,c1,r,f,10000,rng); mc.append(cst.mean()/n); mci.append(ci(cst)/n)
    ax.errorbar(kmc,mc,yerr=mci,fmt=mk,ms=3,color=col,mfc='white',mec=col,lw=.6,capsize=1.2,zorder=3)
    kopt=int(ks[np.argmin(ex)]); kl=law_k(p,c0,c1,r,f)
    ax.axvline(kl,color=col,lw=.6,ls=(0,(2,2)))
    fig2[str(p)]=dict(kopt=kopt,klaw=kl,min_over=float(ex.min()),mc=mc,mc_ci=mci,kmc=kmc,mc_maxdev=float(max(abs(np.array(mc)-np.array([uniform_cost(n,k,p,c0,c1,r,f)[0]/n for k in kmc]))/np.array(mci))))
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([1,2,4,8,16,48]); ax.set_xticklabels(['1','2','4','8','16','48'])
from matplotlib.ticker import NullFormatter
ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks([1.5,2,3,5]); ax.set_yticklabels(['1.5','2','3','5'])
ax.set_xlabel('verification interval k (steps)'); ax.set_ylabel('expected cost per step'); ax.legend(frameon=False,fontsize=6.8,loc='upper left',handlelength=1.5)
ax.set_title('(a) exact cost and simulation',fontsize=7.5,loc='left')
# panel b: k* vs exact optimum over grid
g=pd.DataFrame(json.load(open('results/abstract_A1A2.json'))['A2'])
ax=axs[1]
sel=g[(g.kopt<120)]
ax.scatter(sel.kopt,sel.klaw,s=4,color=BLUE,alpha=.45,lw=0)
hi=sel[sel.pk>=0.3]; ax.scatter(hi.kopt,hi.klaw,s=5,color=ORG,alpha=.8,lw=0,marker='s')
lim=[0.8,70]; ax.plot(lim,lim,color=INK,lw=.6)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xticks([1,3,10,30]); ax.set_xticklabels(['1','3','10','30']); ax.set_yticks([1,3,10,30]); ax.set_yticklabels(['1','3','10','30'])
ax.set_xlabel('exact optimal interval'); ax.set_ylabel('interval from Eq. (7)')
ax.set_title('(b) grid of 756 settings',fontsize=7.5,loc='left')
ax.text(.97,.05,'orange: p k* ≥ 0.3',transform=ax.transAxes,fontsize=6.2,color=ORG,ha='right')
fig.tight_layout(w_pad=1.0); fig.savefig('figs/fig2_law.png',bbox_inches='tight',pad_inches=.03); plt.close(fig)
json.dump(fig2,open('results/fig2_data.json','w'),indent=1)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk in('kopt','klaw','min_over','mc_maxdev')} for k,v in fig2.items()},indent=1))
