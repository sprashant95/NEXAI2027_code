import numpy as np, json, time
from model import *
from abstract_sim import mc_uniform, ci

rng = np.random.default_rng(777)
out = {}

# ---------- A3: heterogeneity, DP vs uniform
n = 48; pbar = 0.03; c0, c1, r, f = 0.5, 0.0, 0.8, 0.05
A3 = []
for cv in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    gaps_u, gaps_law, gaps_every = [], [], []
    for _ in range(1000):
        if cv == 0:
            p = np.full(n, pbar)
        else:
            s2 = np.log(1 + cv ** 2); mu = np.log(pbar) - s2 / 2
            p = np.clip(rng.lognormal(mu, np.sqrt(s2), n), 1e-4, 0.5)
        cuts, cdp = dp_schedule(p, c0, c1, r, f)
        # best uniform for this p vector
        best = np.inf
        for k in range(1, n + 1):
            ed = list(range(k, n, k)) + [n]
            best = min(best, schedule_stats(p, ed, c0, c1, r, f)[0])
        kl = int(round(law_k(p.mean(), c0, c1, r, f)))
        edl = list(range(kl, n, kl)) + [n]
        cl = schedule_stats(p, edl, c0, c1, r, f)[0]
        ce = schedule_stats(p, list(range(1, n + 1)), c0, c1, r, f)[0]
        gaps_u.append(best / cdp - 1); gaps_law.append(cl / cdp - 1); gaps_every.append(ce / cdp - 1)
    A3.append(dict(cv=cv, best_uniform_gap=float(np.mean(gaps_u)), law_gap=float(np.mean(gaps_law)), every_gap=float(np.mean(gaps_every)),
                   law_gap_p95=float(np.quantile(gaps_law, .95)), law_gap_ci=float(ci(np.array(gaps_law)))))
    print('A3', A3[-1])
out['A3'] = A3

# ---------- A4: bursty (Markov-modulated) errors
n = 48; pbar = 0.03; c0, c1, r, f = 0.5, 0.0, 0.8, 0.05
ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 48]
A4 = []
for kappa in [1, 3, 10]:
    curve = []
    for k in ks:
        cost, clean = mc_uniform(n, k, pbar, c0, c1, r, f, 20000, rng, burst=(kappa, 5.0, 0.1) if kappa > 1 else None)
        curve.append(dict(k=k, cost=float(cost.mean()), ci=float(ci(cost)), succ=float(clean.mean())))
    kopt = min(curve, key=lambda d: d['cost'])['k']
    klaw = int(round(law_k(pbar, c0, c1, r, f)))
    ck = {d['k']: d['cost'] for d in curve}
    if klaw not in ck:
        cost, clean = mc_uniform(n, klaw, pbar, c0, c1, r, f, 20000, rng, burst=(kappa, 5.0, 0.1) if kappa > 1 else None)
        ck[klaw] = float(cost.mean())
    A4.append(dict(kappa=kappa, kopt=kopt, klaw=klaw, regret=ck[klaw] / ck[kopt] - 1, curve=curve))
    print('A4 kappa', kappa, 'kopt', kopt, 'klaw', klaw, 'regret %.4f' % A4[-1]['regret'], 'costs', [round(d['cost'], 1) for d in curve])
out['A4'] = A4

# ---------- A5: adaptive controller
grid = np.exp(np.linspace(np.log(1e-4), np.log(0.4), 160))

def ll_seg(L, alarm, r, f):
    q = (1 - grid) ** L
    a = r * (1 - q) + f * q
    return np.log(a) if alarm else np.log1p(-a)

def run_policy(n, p_fn, policy, p0, c0, c1, r, f, rng, gamma=0.9, prior_sd=1.0):
    pos = 0; cost = 0.0; clean = True
    ll = np.zeros_like(grid)
    k_fixed = max(1, int(round(law_k(p0, c0, c1, r, f))))
    while pos < n:
        if policy == 'fixed': k = k_fixed
        elif policy == 'oracle': k = max(1, int(round(law_k(p_fn(pos), c0, c1, r, f))))
        else:
            post = ll - 0.5 * ((np.log(grid) - np.log(p0)) / prior_sd) ** 2
            ph = grid[int(np.argmax(post))]
            k = max(1, int(round(law_k(ph, c0, c1, r, f))))
        L = min(k, n - pos)
        while True:
            cost += L * (1 + c1) + c0
            p_here = np.array([p_fn(pos + i) for i in range(L)])
            err = (rng.random(L) < p_here).any()
            alarm = rng.random() < (r if err else f)
            if policy == 'adaptive': ll = gamma * ll + ll_seg(L, alarm, r, f)
            if not alarm:
                if err: clean = False
                break
        pos += L
    return cost, clean

n = 96; p0 = 0.03; c0, c1, r, f = 0.5, 0.0, 0.8, 0.05
A5 = []
scen = [('x0.1', lambda pos: 0.003), ('x0.25', lambda pos: 0.0075), ('x1', lambda pos: 0.03), ('x4', lambda pos: 0.12), ('x10', lambda pos: 0.3 if False else 0.25),
        ('drift x1->x4', lambda pos: 0.03 if pos < 48 else 0.12), ('drift x1->x0.25', lambda pos: 0.03 if pos < 48 else 0.0075)]
for name, pf in scen:
    row = dict(scenario=name)
    for pol in ['fixed', 'adaptive', 'oracle']:
        res = [run_policy(n, pf, pol, p0, c0, c1, r, f, rng) for _ in range(1500)]
        cs = np.array([x[0] for x in res]); ss = np.array([x[1] for x in res])
        row[pol] = dict(cost=float(cs.mean()), ci=float(ci(cs)), succ=float(ss.mean()))
    A5.append(row)
    print('A5', name, {k: (round(v['cost'], 1), round(v['succ'], 3)) for k, v in row.items() if k != 'scenario'})
out['A5'] = A5
json.dump(out, open('results/abstract_A3A4A5.json', 'w'))
