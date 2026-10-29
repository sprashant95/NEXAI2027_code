import numpy as np, json, time, sys
from model import *

def mc_uniform(n, k, p, c0, c1, r, f, N, rng, burst=None):
    """Event-level Monte Carlo, N independent runs. burst=(kappa, mean_storm_len, storm_frac) makes p Markov-modulated
    with the same long-run mean p."""
    lens = [k] * (n // k) + ([n % k] if n % k else [])
    cost = np.zeros(N); clean = np.ones(N, bool)
    if burst:
        kappa, slen, frac = burst
        # regimes: calm hazard pc, storm hazard kappa*pc, long-run mean p
        pc = p / (1 - frac + frac * kappa); ps = kappa * pc
        leave_storm = 1.0 / slen
        enter_storm = leave_storm * frac / (1 - frac)
        storm = rng.random(N) < frac
    for L in lens:
        active = np.ones(N, bool)
        while active.any():
            idx = np.flatnonzero(active)
            cost[idx] += L * (1 + c1) + c0
            if burst:
                err = np.zeros(idx.size, bool)
                st = storm[idx].copy()
                for _ in range(L):
                    h = np.where(st, ps, pc)
                    err |= rng.random(idx.size) < h
                    u = rng.random(idx.size)
                    st = np.where(st, u > leave_storm, u < enter_storm)
                storm[idx] = st
            else:
                err = (rng.random((idx.size, L)) < p).any(axis=1)
            u = rng.random(idx.size)
            alarm = np.where(err, u < r, u < f)
            ok = ~alarm
            done = idx[ok]
            clean[done[err[ok]]] = False
            active[done] = False
    return cost, clean

def ci(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

if __name__ == "__main__":
    rng = np.random.default_rng(20271)
    out = {}
    t0 = time.time()
    # ---- A1: validate exact formulas against event-level Monte Carlo
    rows = []
    n = 48
    for (p, c0, c1, r, f) in [(0.01, 0.5, 0.0, 0.8, 0.0), (0.03, 0.5, 0.1, 0.8, 0.05), (0.08, 1.0, 0.2, 0.6, 0.1), (0.03, 0.25, 0.0, 0.95, 0.02)]:
        for k in [1, 2, 4, 8, 16, 48]:
            cost, clean = mc_uniform(n, k, p, c0, c1, r, f, 20000, rng)
            ec, es = uniform_cost(n, k, p, c0, c1, r, f)
            rows.append(dict(p=p, c0=c0, c1=c1, r=r, f=f, k=k, mc_cost=float(cost.mean()), mc_ci=float(ci(cost)), exact_cost=ec,
                             mc_succ=float(clean.mean()), mc_succ_ci=float(1.96*np.sqrt(clean.mean()*(1-clean.mean())/len(clean))), exact_succ=es))
    out['A1'] = rows
    z = [abs(x['mc_cost'] - x['exact_cost']) / x['mc_ci'] for x in rows]
    zs = [abs(x['mc_succ'] - x['exact_succ']) / max(x['mc_succ_ci'],1e-9) for x in rows]
    print('A1 cost max |dev|/CI95 = %.2f, share within CI %.2f; succ share within CI %.2f; max rel cost dev %.4f' % (max(z), np.mean(np.array(z) <= 1), np.mean(np.array(zs) <= 1),
          max(abs(x['mc_cost'] / x['exact_cost'] - 1) for x in rows)))
    # ---- A2: regret of closed-form interval over a grid
    grid = []
    for p in [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
            for r in [0.5, 0.7, 0.9, 0.99]:
                for f in [0.0, 0.05, 0.15]:
                    if r_eff(r, f) <= 0: continue
                    n = 120
                    kopt, copt, sopt = best_uniform(n, p, c, 0.0, r, f)
                    kl = max(1, int(round(law_k(p, c, 0.0, r, f))))
                    kl = min(kl, n)
                    cl, sl = uniform_cost(n, kl, p, c, 0.0, r, f)
                    k_se = max(1, min(n, int(round(np.sqrt(c / p)))))   # recall-blind silent-error rule
                    k_yd = max(1, min(n, int(round(np.sqrt(2 * c / p)))))  # classic fail-stop Young rule
                    c_se = uniform_cost(n, k_se, p, c, 0.0, r, f)[0]
                    c_yd = uniform_cost(n, k_yd, p, c, 0.0, r, f)[0]
                    c1k = uniform_cost(n, 1, p, c, 0.0, r, f)[0]
                    cn = uniform_cost(n, n, p, c, 0.0, r, f)[0]
                    grid.append(dict(p=p, c=c, r=r, f=f, kopt=kopt, klaw=kl, k_se=k_se, k_yd=k_yd,
                                     reg_law=cl / copt - 1, reg_se=c_se / copt - 1, reg_yd=c_yd / copt - 1,
                                     reg_every=c1k / copt - 1, reg_end=cn / copt - 1, pk=p * law_k(p, c, 0.0, r, f),
                                     ovh_pred=overhead_law(p, c, 0.0, r, f), ovh_exact=copt / n - 1))
    out['A2'] = grid
    import pandas as pd
    g = pd.DataFrame(grid)
    print('A2 cells', len(g))
    for col in ['reg_law', 'reg_se', 'reg_yd', 'reg_every', 'reg_end']:
        print(col, 'median %.4f  p95 %.4f  max %.4f' % (g[col].median(), g[col].quantile(.95), g[col].max()))
    print('share law regret<1%%: %.3f <2%%: %.3f <5%%: %.3f' % ((g.reg_law < .01).mean(), (g.reg_law < .02).mean(), (g.reg_law < .05).mean()))
    print('law regret when pk*<0.3: median %.4f max %.4f (n=%d); pk*>=0.3: median %.4f max %.4f (n=%d)' % (
        g[g.pk < .3].reg_law.median(), g[g.pk < .3].reg_law.max(), (g.pk < .3).sum(), g[g.pk >= .3].reg_law.median(), g[g.pk >= .3].reg_law.max(), (g.pk >= .3).sum()))
    print('overhead pred vs exact: median abs err %.4f' % (g.ovh_pred - g.ovh_exact).abs().median())
    json.dump(out, open('results/abstract_A1A2.json', 'w'))
    print('time %.1fs' % (time.time() - t0))
