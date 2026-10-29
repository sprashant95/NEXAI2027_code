import sys; sys.path.insert(0,'.')
import json, numpy as np, pandas as pd
from model import *
df=pd.read_csv('results/testbed_runs.csv'); CAL=json.load(open('results/calibration.json'))
def tp(t):
    c=CAL['cost'][t]; L4=CAL['tiers'][f'{t}_L4']; return c['c0'],c['c1'],L4['recall'],L4['fa']
rng=np.random.default_rng(5)
def cps(g, B=1000):
    u=g.units.values; ok=g.ok.values.astype(float)
    est=u.mean()/max(ok.mean(),1e-9)
    bs=[]
    n=len(u)
    for _ in range(B):
        i=rng.integers(0,n,n); s=ok[i].mean()
        bs.append(u[i].mean()/s if s>0 else np.inf)
    return est,np.percentile(bs,2.5),np.percentile(bs,97.5)
rows=[]
for (sc,tier,pol),g in df.groupby(['scale','tier','policy']):
    e,lo,hi=cps(g,400)
    rows.append(dict(scale=sc,tier=tier,policy=pol,units=g.units.mean(),succ=g.ok.mean(),cps=e,lo=lo,hi=hi,wall_ms=g.wall.mean()*1e3,kmean=g.kmean.mean(),nalarm=g.nalarm.mean()))
R=pd.DataFrame(rows); R.to_csv('results/testbed_summary.csv',index=False)
pd.set_option('display.width',200); pd.set_option('display.max_rows',500)
for sc in [1.0]:
    print('=== scale',sc)
    print(R[(R.scale==sc)].sort_values(['tier','policy'])[['tier','policy','kmean','units','succ','cps','lo','hi','wall_ms']].round(3).to_string())
print(R[R.tier=='none'][['scale','units','succ','cps','lo','hi']])
