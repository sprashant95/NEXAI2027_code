import sys; sys.path.insert(0, '.')
import json, numpy as np, pandas as pd
from model import *
df = pd.read_csv('results/testbed_runs.csv'); CAL = json.load(open('results/calibration.json'))
rng = np.random.default_rng(2027)
B = 2000
N = 48
def tp(t):
    c = CAL['cost'][t]; L4 = CAL['tiers'][f'{t}_L4']; return c['c0'], c['c1'], L4['recall'], L4['fa']

def arr(scale, tier, pol):
    g = df[(df.scale == scale) & (df.tier == tier) & (df.policy == pol)].sort_values('run')
    return g.units.values, g.ok.values.astype(float), g

def boot_idx(n): return rng.integers(0, n, (B, n))
IDX = boot_idx(400)
def gamma(u, ok, idx=None):
    return u.mean() / max(ok.mean(), 1e-12)
def gamma_ci(u, ok):
    s = ok[IDX].mean(1); m = u[IDX].mean(1); bs = np.where(s > 0, m / np.maximum(s, 1e-12), np.inf)
    return gamma(u, ok), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def ratio_ci(a, b):
    """Gamma_a / Gamma_b with paired bootstrap over run index."""
    (ua, oa), (ub, ob) = a, b
    ga = ua[IDX].mean(1) / np.maximum(oa[IDX].mean(1), 1e-12); gb = ub[IDX].mean(1) / np.maximum(ob[IDX].mean(1), 1e-12)
    r = ga / gb
    return float(gamma(ua, oa) / gamma(ub, ob)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24]
out = {'rows': {}, 'plan': {}}
pm = CAL['p_eff_mean']
for scale in [0.5, 1.0, 2.0]:
    pbar = pm * scale
    un, on, _ = arr(scale, 'none', 'none'); gn = gamma_ci(un, on)
    out['rows'][f'{scale}|none'] = dict(units=float(un.mean()), succ=float(on.mean()), gamma=gn)
    for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
        c0, c1, r, f = tp(t)
        pols = ['end'] + [f'k{k}' for k in GRID] + ['law', 'se', 'dp']
        res = {}
        for pol in pols:
            u, o, g = arr(scale, t, pol)
            res[pol] = dict(units=float(u.mean()), succ=float(o.mean()), gamma=gamma_ci(u, o), kmean=float(g.kmean.mean()), nalarm=float(g.nalarm.mean()),
                            wall_ms=float(g.wall.mean() * 1e3), u=u, o=o)
        kl = law_k(pbar, c0, c1, r, f); klr = int(min(N, max(1, round(kl)))) if np.isfinite(kl) else N
        gridg = {k: res[f'k{k}']['gamma'][0] for k in GRID}; kbest = min(gridg, key=gridg.get)
        # include end-only as k=48 candidate for 'best fixed'
        cands = dict(gridg); cands[48] = res['end']['gamma'][0]; kbest = min(cands, key=cands.get)
        bestpol = 'end' if kbest == 48 else f'k{kbest}'
        law = (res['law']['u'], res['law']['o'])
        rel = {}
        for pol in pols:
            if pol == 'law': continue
            rel[pol] = ratio_ci((res[pol]['u'], res[pol]['o']), law)
        rel['none'] = ratio_ci((un, on), law)
        rel['best_fixed'] = ratio_ci((res[bestpol]['u'], res[bestpol]['o']), law)
        # model predictions for law schedule
        ec, es = uniform_cost(N, klr, pbar, c0, c1, r, f)
        pred_gamma_exact = ec / es
        gh = N * (1 + c1) / (1 - f) * (1 + 2 * np.sqrt(max(r_eff(r, f), 0) * pbar * (c0 / (1 + c1)))) * np.exp(N * pbar * (1 - r) / (1 - f))
        plan = dict(scale=scale, tier=t, pbar=pbar, c0=c0, c1=c1, r=r, f=f, klaw=float(kl), klaw_int=klr, kbest=kbest, pred_cost=ec, pred_succ=es, pred_gamma_exact=pred_gamma_exact,
                    pred_gamma_eq9=float(gh), meas_cost=res['law']['units'], meas_succ=res['law']['succ'], meas_gamma=res['law']['gamma'],
                    ovh_pred=float(overhead_law(pbar, c0, c1, r, f)))
        out['plan'][f'{scale}|{t}'] = plan
        for pol in res: res[pol].pop('u'); res[pol].pop('o')
        out['rows'][f'{scale}|{t}'] = dict(res=res, rel=rel)
        print(scale, t, 'klaw %.2f kbest %d' % (kl, kbest), 'Gamma law %.1f [%.1f,%.1f]' % res['law']['gamma'], 'best fixed/law %.3f' % rel['best_fixed'][0],
              'every/law %.2f end/law %.2f none/law %.2f dp/law %.3f se/law %.3f' % (rel['k1'][0], rel['end'][0], rel['none'][0], rel['dp'][0], rel['se'][0]),
              '| pred G9 %.1f exact %.1f' % (gh, pred_gamma_exact))
# wall clock vs units
w = df[df.tier.isin(['T1', 'T2', 'T3', 'T4']) | (df.tier == 'none')]
w = w[w.tier != 'T5']
slope = np.polyfit(w.units, w.wall * 1e3, 1)
r2 = np.corrcoef(w.units, w.wall)[0, 1] ** 2
out['wall'] = dict(slope_ms_per_unit=float(slope[0]), intercept=float(slope[1]), r2=float(r2), t_step_ms=CAL['t_step_us'] / 1e3, n=int(len(w)))
print('wall vs units: slope %.4f ms/unit (t_step %.4f ms) r2 %.3f n=%d' % (slope[0], CAL['t_step_us'] / 1e3, r2, len(w)))
json.dump(out, open('results/summary.json', 'w'), indent=1, default=float)
