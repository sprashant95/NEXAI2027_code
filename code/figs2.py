import sys; sys.path.insert(0,'.')
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
                     'xtick.major.width':.6,'ytick.major.width':.6,'savefig.dpi':600,'figure.dpi':150})
S = json.load(open('results/summary.json'))
BLUE,ORG,VIO,GRN,GRY,INK,MUT='#2a78d6','#eb6834','#6250d6','#008300','#8a8984','#1c1c1c','#6b6a66'
STY = {'T1':(GRY,'v',(0,(1,1.5))), 'T2':(BLUE,'o','-'), 'T3':(ORG,'s','-'), 'T4':(VIO,'^','-'), 'T5':(GRN,'D',(0,(4,2)))}
NAME = {'T1':'T1 structural','T2':'T2 replay 40','T3':'T3 replay 200','T4':'T4 replay all','T5':'T5 emulated judge'}
W=4.7
KS=[1,2,3,4,6,8,12,16,24,48]
def curve(scale,t,key):
    r=S['rows'][f'{scale}|{t}']['res']; out=[]
    for k in KS:
        d=r['end' if k==48 else f'k{k}']; out.append(d[key])
    return out

# ---------------- Fig 3
fig,axs=plt.subplots(1,2,figsize=(W,2.25))
ax=axs[0]
for t in ['T1','T2','T3','T4','T5']:
    col,mk,ls=STY[t]; g=curve(1.0,t,'gamma'); est=np.array([x[0] for x in g]); lo=np.array([x[1] for x in g]); hi=np.array([x[2] for x in g])
    ax.plot(KS,est,color=col,ls=ls,lw=1.0,marker=mk,ms=2.8,mfc='white' if t!='T3' else col,mec=col,label=NAME[t],zorder=3)
    ax.fill_between(KS,lo,hi,color=col,alpha=.13,lw=0)
    kl=S['plan'][f'1.0|{t}']['klaw']
    if np.isfinite(kl): ax.plot([kl],[np.interp(np.log(kl),np.log(KS),est)],marker='|',ms=9,mew=1.4,color=col,zorder=4)
gn=S['rows']['1.0|none']['gamma'][0]
ax.axhline(gn,color=INK,lw=.7,ls=(0,(5,2))); ax.text(1.05,gn*1.03,'no verification',fontsize=6.2,color=INK,va='bottom')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([1,2,4,8,16,48]); ax.set_xticklabels(['1','2','4','8','16','end'])
ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks([100,150,250,400]); ax.set_yticklabels(['100','150','250','400'])
ax.set_xlabel('verification interval k (steps)'); ax.set_ylabel('cost per correct completion'); ax.set_title('(a) cost',fontsize=7.5,loc='left')
ax=axs[1]
for t in ['T1','T2','T3','T4','T5']:
    col,mk,ls=STY[t]; sc=[x for x in curve(1.0,t,'succ')]
    ax.plot(KS,sc,color=col,ls=ls,lw=1.0,marker=mk,ms=2.8,mfc='white' if t!='T3' else col,mec=col,zorder=3)
ax.axhline(S['rows']['1.0|none']['succ'],color=INK,lw=.7,ls=(0,(5,2)))
ax.set_xscale('log'); ax.set_xticks([1,2,4,8,16,48]); ax.set_xticklabels(['1','2','4','8','16','end']); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0,1.03); ax.set_xlabel('verification interval k (steps)'); ax.set_ylabel('success rate'); ax.set_title('(b) success',fontsize=7.5,loc='left')
h_,l_=axs[0].get_legend_handles_labels(); ax.legend(h_,l_,ncol=2,frameon=False,fontsize=5.6,loc='center',bbox_to_anchor=(.5,.55),handlelength=1.8,columnspacing=.8,labelspacing=.3)
fig.tight_layout(w_pad=1.2); fig.savefig('figs/fig3_testbed.png',bbox_inches='tight',pad_inches=.03); plt.close(fig)

# ---------------- Fig 4
fig,axs=plt.subplots(1,3,figsize=(W,2.0),sharey=True)
for ax,sc in zip(axs,[0.5,1.0,2.0]):
    xs=np.arange(5); tiers=['T1','T2','T3','T4','T5']
    for x,t in zip(xs,tiers):
        col,mk,ls=STY[t]; p=S['plan'][f'{sc}|{t}']; g=p['meas_gamma']
        ax.errorbar([x],[g[0]],yerr=[[g[0]-g[1]],[g[2]-g[0]]],fmt=mk,ms=4,color=col,mfc=col,mec=col,lw=.8,capsize=1.5,zorder=3)
        ax.plot([x],[p['pred_gamma_eq9']],marker='x',ms=4.5,mew=1.0,color=INK,ls='none',zorder=4)
    ax.axhline(S['rows'][f'{sc}|none']['gamma'][0],color=INK,lw=.7,ls=(0,(5,2)))
    ax.set_yscale('log'); ax.set_xticks(xs); ax.set_xticklabels(['T1','T2','T3','T4','T5'],fontsize=6.5); ax.set_xlim(-.6,4.6)
    ax.set_title(f'm = {sc}',fontsize=7.5,loc='left'); ax.yaxis.set_minor_formatter(NullFormatter())
axs[0].set_ylabel('cost per correct completion')
lo=min(min(S['plan'][f'{s}|{t}']['meas_gamma'][1] for t in ['T1','T2','T3','T4','T5']) for s in [0.5,1.0,2.0])
hi=max(S['rows'][f'{s}|none']['gamma'][0] for s in [0.5,1.0,2.0])
axs[0].set_ylim(lo*.8,hi*1.25)
ticks=[t for t in [20,30,50,100,200,500,1000] if lo*.8<=t<=hi*1.25]
axs[0].set_yticks(ticks); axs[0].set_yticklabels([str(t) for t in ticks])
axs[2].text(.98,.97,'dot: measured, 95% interval\ncross: prediction, Eq. (9)\ndashed: no verification',transform=axs[2].transAxes,fontsize=5.8,va='top',ha='right',color=MUT)
fig.tight_layout(w_pad=.6); fig.savefig('figs/fig4_selection.png',bbox_inches='tight',pad_inches=.03); plt.close(fig)
print('figs ok')
