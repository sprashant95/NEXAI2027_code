import sys, json, time, itertools
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from multiprocessing import Pool
from testbed import *
from model import law_k, dp_schedule, r_eff

CAL = json.load(open('results/calibration.json'))
N = 48; NPIPE = 100; RUNS = 400
SCALES = [0.5, 1.0, 2.0]
TIER_LIST = ['T1', 'T2', 'T3', 'T4', 'T5']
GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24]
_cache = {}

def tier_par(t):
    c = CAL['cost'][t]; L4 = CAL['tiers'][f'{t}_L4']
    return dict(c0=c['c0'], c1=c['c1'], r=L4['recall'], f=L4['fa'])

def get_pipe(j):
    if j not in _cache:
        ops = make_pipeline(5000 + j); d0 = make_data(7000 + j); g = run_ref(ops, d0)
        _cache[j] = (ops, d0, g)
    return _cache[j]

def cuts_uniform(k, n=N):
    return list(range(k, n, k)) + [n]

import os, pickle
def task(args):
    scale, tier, pol = args
    part = f'results/parts/{scale}_{tier}_{pol}.pkl'
    if os.path.exists(part): return pickle.load(open(part, 'rb'))
    rows = []
    tp = tier_par(tier) if tier else None
    pbar = CAL['p_eff_mean'] * scale
    for run in range(RUNS):
        j = run % NPIPE; seed = 91000 + run
        ops, d0, g = get_pipe(j)
        if pol == 'none': cuts = [N]
        elif pol == 'end': cuts = [N]
        elif pol.startswith('k'): cuts = cuts_uniform(int(pol[1:]))
        elif pol in ('law', 'se'):
            if pol == 'law': k = law_k(pbar, tp['c0'], tp['c1'], tp['r'], tp['f'])
            else: k = np.sqrt((tp['c0'] / (1 + tp['c1'])) / pbar)
            k = N if not np.isfinite(k) else int(min(N, max(1, round(k)))); cuts = cuts_uniform(k)
        elif pol == 'dp':
            p_i = np.array([CAL['p_eff'][o['kind']] * scale for o in ops])
            cuts, _ = dp_schedule(p_i, tp['c0'], tp['c1'], tp['r'], tp['f'])
        t_arg = None if pol == 'none' else tier
        c0, c1 = (tp['c0'], tp['c1']) if tp else (0.0, 0.0)
        res = run_workflow(ops, d0, g, cuts, t_arg, seed, scale, c0, c1)
        res.update(scale=scale, tier=tier if pol != 'none' else 'none', policy=pol, run=run, ncut=len(cuts), kmean=N / len(cuts))
        rows.append(res)
    pickle.dump(rows, open(part, 'wb'))
    return rows

if __name__ == '__main__':
    t0 = time.time(); os.makedirs('results/parts', exist_ok=True)
    tasks = []
    for s in SCALES:
        tasks.append((s, None, 'none'))
        for t in TIER_LIST:
            for pol in ['end'] + [f'k{k}' for k in GRID] + ['law', 'se', 'dp']:
                tasks.append((s, t, pol))
    # run heavier tiers first for balance
    tasks.sort(key=lambda x: {'T4': 0, None: 9}.get(x[1], 1))
    allrows = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
            allrows.extend(rows)
            if i % 10 == 0: print(i + 1, '/', len(tasks), '%.0fs' % (time.time() - t0), flush=True)
    df = pd.DataFrame(allrows); df.to_csv('results/testbed_runs.csv', index=False)
    print('saved', len(df), 'rows in %.0fs' % (time.time() - t0))
