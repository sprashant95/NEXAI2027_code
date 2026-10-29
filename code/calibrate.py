import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from testbed import *

rng = np.random.default_rng(424242)
NCAL = 60
t0 = time.time()
pipes = []
for s in range(1, NCAL + 1):
    ops = make_pipeline(s)
    d0 = make_data(100 + s)
    pipes.append((ops, d0))

# golden prefix states
def prefix_states(ops, d0):
    sts = [d0]
    for op in ops: sts.append(ref_op(sts[-1], op))
    return sts

pre = [prefix_states(o, d) for o, d in pipes]

# ---- effective failure probability per kind
eff = {k: [0, 0] for k in KINDS}
eff_ft = {f: [0, 0] for f in FAULTS}
records = []   # (tier-detect flags computed below)
t1 = time.time()
for (ops, d0), sts in zip(pipes, pre):
    for i, op in enumerate(ops):
        for _ in range(6):
            ft = FAULTS[int(rng.choice(len(FAULTS), p=FAULT_W))]
            new = faulty_op(sts[i], op, ft, rng)
            ref = sts[i + 1]
            e = effective(new, ref)
            eff[op['kind']][0] += int(e); eff[op['kind']][1] += 1
            eff_ft[ft][0] += int(e); eff_ft[ft][1] += 1
p_eff = {k: P_INJ[k] * eff[k][0] / eff[k][1] for k in KINDS}
print('effective share by kind', {k: round(v[0] / v[1], 3) for k, v in eff.items()})
print('effective share by fault', {k: round(v[0] / v[1], 3) for k, v in eff_ft.items()})
# mean per-step p_eff for the op mix actually used in pipelines
pm = np.mean([p_eff[op['kind']] for ops, _ in pipes for op in ops])
print('mean p_eff per step (scale 1): %.4f ; mean p_inj: %.4f' % (pm, np.mean([P_INJ[op['kind']] for ops, _ in pipes for op in ops])))

# ---- recall and false alarms, by tier and segment length
vr = np.random.default_rng(99)
res = {}
for L in [1, 4, 12]:
    for tier in TIERS:
        det = 0; n_eff = 0; fa = 0; n_clean = 0
        det_by_ft = {f: [0, 0] for f in FAULTS}
        for pi, ((ops, d0), sts) in enumerate(zip(pipes, pre)):
            for rep in range(40 if L == 1 else 25):
                s0 = int(rng.integers(0, len(ops) - L + 1)); seg = ops[s0:s0 + L]
                prev = sts[s0]; ref = sts[s0 + L]
                # clean
                fa += int(verify(tier, prev, ref, seg, vr)); n_clean += 1
                # one failed step inside the segment
                j = int(rng.integers(0, L)); ft = FAULTS[int(rng.choice(len(FAULTS), p=FAULT_W))]
                st = sts[s0]
                for q, op in enumerate(seg):
                    st = faulty_op(st, op, ft, rng) if q == j else ref_op(st, op)
                if effective(st, ref):
                    d = verify(tier, prev, st, seg, vr)
                    det += int(d); n_eff += 1; det_by_ft[ft][0] += int(d); det_by_ft[ft][1] += 1
        res[f'{tier}_L{L}'] = dict(recall=det / n_eff, n_eff=n_eff, fa=fa / n_clean, n_clean=n_clean,
                                   by_fault={f: (v[0] / v[1] if v[1] else None) for f, v in det_by_ft.items()})
        print(tier, 'L', L, 'recall %.3f (n=%d) false-alarm %.4f (n=%d)' % (det / n_eff, n_eff, fa / n_clean, n_clean), flush=True)

# ---- timing: step time and verification time as a + b*L
def tmed(fn, reps):
    ts = []
    for _ in range(reps):
        t = perf_counter(); fn(); ts.append(perf_counter() - t)
    return float(np.median(ts))
step_times = []; ver = {tier: {} for tier in TIERS}
for L in [1, 2, 4, 8, 16]:
    st_l = []; vt = {tier: [] for tier in TIERS}
    for (ops, d0), sts in list(zip(pipes, pre))[:30]:
        for _ in range(6):
            s0 = int(rng.integers(0, len(ops) - L + 1)); seg = ops[s0:s0 + L]; prev = sts[s0]
            def f_seg():
                st = prev
                for op in seg: st = ref_op(st, op)
            st_l.append(tmed(f_seg, 15) / L)
            new = sts[s0 + L]
            for tier in TIERS:
                vt[tier].append(tmed(lambda: verify(tier, prev, new, seg, vr), 15) if tier != 'T5' else 0.0)
    step_times.extend(st_l)
    for tier in TIERS: ver[tier][L] = float(np.mean(vt[tier]))
for L in ver['T5']: ver['T5'][L] = None
t_step = float(np.mean(step_times))
print('mean step time %.1f us' % (t_step * 1e6))
cost = {}
Ls = np.array([1, 2, 4, 8, 16], float)
for tier in TIERS:
    if tier == 'T5':
        cost[tier] = dict(c0=1.0, c1=0.0, raw_us=None); print('T5 assumed c0=1.0 c1=0'); continue
    y = np.array([ver[tier][int(l)] for l in Ls])
    A = np.vstack([np.ones_like(Ls), Ls]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    cost[tier] = dict(c0=float(a / t_step), c1=float(b / t_step), raw_us={int(l): ver[tier][int(l)] * 1e6 for l in Ls})
    print(tier, 'c0 %.3f  c1 %.3f' % (a / t_step, b / t_step), {int(l): round(ver[tier][int(l)] * 1e6, 1) for l in Ls})
json.dump(dict(p_eff=p_eff, p_inj=P_INJ, eff_share_kind={k: v[0] / v[1] for k, v in eff.items()}, eff_share_fault={k: v[0] / v[1] for k, v in eff_ft.items()},
               p_eff_mean=pm, tiers=res, cost=cost, t_step_us=t_step * 1e6, ncal=NCAL), open('results/calibration.json', 'w'), indent=1)
print('done %.0fs' % (time.time() - t0))
