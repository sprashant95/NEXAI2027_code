"""Fault-injection testbed: executable data-processing workflows whose steps fail silently.
Every step is a real numpy transformation. A failed step returns a plausible but wrong result.
Verifiers are local: they see the segment's input checkpoint, its output, and the segment's specification.
"""
import numpy as np, time
from time import perf_counter

M = 20000
NG = 24
KINDS = ['scale', 'shift', 'clip', 'affine2', 'logabs', 'tanh', 'ratio', 'lookup', 'bucket', 'round', 'remap', 'filter']
KIND_W = np.array([.08, .08, .08, .14, .08, .08, .08, .10, .06, .06, .04, .04]); KIND_W = KIND_W / KIND_W.sum()
# per-kind injected failure probability per execution of a step
P_INJ = dict(scale=.015, shift=.015, clip=.03, affine2=.04, logabs=.02, tanh=.02, ratio=.04, lookup=.06, bucket=.04,
             round=.015, remap=.05, filter=.05)
FAULTS = ['const', 'wrong_col', 'subset', 'nan', 'drop', 'skip']
FAULT_W = np.array([.20, .15, .30, .10, .10, .15])

class State:
    __slots__ = ('ids', 'cols')
    def __init__(s, ids, cols): s.ids = ids; s.cols = cols

def make_data(seed):
    rng = np.random.default_rng(seed)
    cols = {'a': rng.normal(0, 1, M), 'b': rng.lognormal(0, .5, M), 'c': rng.uniform(-2, 2, M), 'd': rng.normal(5, 2, M),
            'g': rng.integers(0, NG, M).astype(np.int64)}
    cols['h'] = (rng.integers(0, 1000, M) % NG).astype(np.int64)
    return State(np.arange(M, dtype=np.int64), cols)

def compute(op, cols):
    k = op['kind']
    if k == 'scale': return cols[op['in1']] * op['k']
    if k == 'shift': return cols[op['in1']] + op['k']
    if k == 'clip': return np.clip(cols[op['in1']], op['lo'], op['hi'])
    if k == 'affine2': return op['al'] * cols[op['in1']] + op['be'] * cols[op['in2']]
    if k == 'logabs': return np.log1p(np.abs(cols[op['in1']]))
    if k == 'tanh': return np.tanh(cols[op['in1']] / op['s'])
    if k == 'ratio': return cols[op['in1']] / (1 + np.abs(cols[op['in2']]))
    if k == 'lookup': return op['tab'][cols[op['key']]]
    if k == 'bucket': return np.digitize(cols[op['in1']], op['edges']).astype(np.int64)
    if k == 'round': return np.round(cols[op['in1']], op['d'])
    if k == 'remap': return op['perm'][cols[op['key']]]
    raise ValueError(k)

def ref_op(state, op):
    if op['kind'] == 'filter':
        mask = state.cols[op['in1']] > op['thr']
        return State(state.ids[mask], {c: v[mask] for c, v in state.cols.items()})
    cols = dict(state.cols); cols[op['out']] = compute(op, state.cols)
    return State(state.ids, cols)

def make_pipeline(seed, n=48):
    rng = np.random.default_rng(seed)
    while True:
        st = make_data(seed + 10**6)
        ops = []; floats = ['a', 'b', 'c', 'd']; ints = ['g', 'h']; nfilt = 0; ok = True
        for i in range(n):
            kind = KINDS[rng.choice(len(KINDS), p=KIND_W)]
            if kind == 'filter' and nfilt >= 3: kind = 'scale'
            if kind == 'filter': nfilt += 1
            w = np.arange(1, len(floats) + 1, dtype=float); w /= w.sum()
            in1 = floats[rng.choice(len(floats), p=w)]; in2 = floats[rng.choice(len(floats), p=w)]
            op = dict(kind=kind, in1=in1, in2=in2, idx=i)
            overwrite = (rng.random() < .3) and len(floats) > 4
            outname = (floats[4 + rng.integers(0, len(floats) - 4)] if overwrite else f'x{i}')
            if kind == 'scale': op['k'] = float(rng.uniform(.7, 1.4))
            elif kind == 'shift': op['k'] = float(rng.uniform(-1, 1))
            elif kind == 'clip':
                col = st.cols[in1]; op['lo'] = float(np.quantile(col, .02)); op['hi'] = float(np.quantile(col, .98))
            elif kind == 'affine2': op['al'] = float(rng.uniform(-1, 1)); op['be'] = float(rng.uniform(-.6, .6))
            elif kind == 'tanh': op['s'] = float(rng.uniform(.5, 3))
            elif kind == 'lookup':
                op['tab'] = rng.normal(0, 1, NG); op['key'] = ints[rng.integers(0, len(ints))]
            elif kind == 'bucket':
                nb = int(rng.integers(4, 9)); op['edges'] = np.quantile(st.cols[in1], np.linspace(.1, .9, nb)); op['range'] = nb + 1
            elif kind == 'round': op['d'] = int(rng.integers(2, 5))
            elif kind == 'remap': op['perm'] = rng.permutation(NG).astype(np.int64); op['key'] = ints[rng.integers(0, len(ints))]
            elif kind == 'filter': op['thr'] = float(np.quantile(st.cols[in1], .012))
            # outputs
            if kind == 'filter': op['out'] = None; op['dtype'] = None
            elif kind in ('bucket', 'remap'):
                op['out'] = f'i{i}'; op['dtype'] = 'int64'
                if kind == 'remap': op['range'] = NG
            else:
                op['out'] = outname; op['dtype'] = 'float64'
                if outname not in floats: floats.append(outname)
            if kind == 'lookup': op['out'] = outname
            st = ref_op(st, op)
            for c in st.cols.values():
                if c.dtype.kind == 'f' and (not np.isfinite(c).all() or np.abs(c).max() > 1e9): ok = False
            ops.append(op)
        if ok and len(st.ids) > .9 * M:
            return ops
        seed += 7919; rng = np.random.default_rng(seed)

def run_ref(ops, state):
    for op in ops: state = ref_op(state, op)
    return state

def states_equal(a, b):
    if not np.array_equal(a.ids, b.ids): return False
    if a.cols.keys() != b.cols.keys(): return False
    return all(np.array_equal(a.cols[c], b.cols[c], equal_nan=True) for c in a.cols)

def _alt_col(state, op, rng, key=False):
    if key: cand = [c for c in ('g', 'h') if c != op.get('key')]
    else:
        cand = [c for c, v in state.cols.items() if v.dtype.kind == 'f' and c not in (op['in1'], op.get('in2'))]
    return cand[int(rng.integers(0, len(cand)))] if cand else None

def faulty_op(state, op, ftype, rng):
    """Return the state a failed tool call would leave behind."""
    k = op['kind']; ids = state.ids
    ms = int(rng.choice([200, 50, 10, 4], p=[.25, .3, .3, .15])); r0 = int(rng.integers(0, ms)); delta = float(rng.choice([-1, 1]) * rng.uniform(.03, .3))
    if k == 'filter':
        if ftype in ('const',):
            o2 = dict(op); o2['thr'] = op['thr'] + float(rng.choice([-1, 1]) * rng.uniform(.2, .6)); return ref_op(state, o2)
        if ftype == 'wrong_col':
            alt = _alt_col(state, op, rng)
            if alt is None: return ref_op(state, op)
            o2 = dict(op); o2['in1'] = alt; return ref_op(state, o2)
        if ftype == 'skip': return State(ids, dict(state.cols))
        mdrop = int(rng.choice([50, 10], p=[.6, .4]))
        s2 = ref_op(state, op); keep = (s2.ids % mdrop) != int(rng.integers(0, mdrop))
        return State(s2.ids[keep], {c: v[keep] for c, v in s2.cols.items()})
    out = op['out']; isint = op['dtype'] == 'int64'; rngv = op.get('range', NG)
    if ftype == 'const' or ftype == 'wrong_col' and ((k in ('lookup', 'remap') and _alt_col(state, op, rng, True) is None)
                                                      or (k not in ('lookup', 'remap') and _alt_col(state, op, rng) is None)):
        y = compute(op, state.cols)
        y = ((y + 1) % rngv).astype(np.int64) if isint else y * (1 + delta)
        cols = dict(state.cols); cols[out] = y; return State(ids, cols)
    if ftype == 'wrong_col':
        o2 = dict(op)
        if k in ('lookup', 'remap'): o2['key'] = _alt_col(state, op, rng, True)
        else:
            o2['in1'] = _alt_col(state, op, rng)
            if k in ('affine2', 'ratio'): o2['in2'] = _alt_col(state, o2, rng) or op['in2']
        y = compute(o2, state.cols); cols = dict(state.cols); cols[out] = y; return State(ids, cols)
    if ftype == 'skip':
        src = state.cols[op['key']] if k in ('lookup', 'remap') else state.cols[op['in1']]
        cols = dict(state.cols); cols[out] = src.copy(); return State(ids, cols)
    y = compute(op, state.cols).copy(); mask = (ids % ms) == r0
    isint = y.dtype.kind in 'iu'   # a corrupted upstream column can change dtype
    if ftype == 'subset':
        y[mask] = ((y[mask] + 1) % rngv) if isint else y[mask] * (1 + delta)
        cols = dict(state.cols); cols[out] = y; return State(ids, cols)
    if ftype == 'nan':
        mask = (ids % int(rng.choice([200, 20]))) == r0 % 20
        if isint: y[mask] = -1
        else: y[mask] = np.nan
        cols = dict(state.cols); cols[out] = y; return State(ids, cols)
    if ftype == 'drop':
        md = int(rng.choice([50, 10], p=[.6, .4])); keep = (ids % md) != int(rng.integers(0, md))
        cols = dict(state.cols); cols[out] = y
        return State(ids[keep], {c: v[keep] for c, v in cols.items()})
    raise ValueError(ftype)

def effective(new, ref):
    return not states_equal(new, ref)

# ---------------- verifiers (local, deterministic given the sample) -----------------
TIERS = {'T1': None, 'T2': 40, 'T3': 200, 'T4': 'all', 'T5': 200}
JUDGE_MISS, JUDGE_FALSE = 0.10, 0.06   # T5 emulates a model-based judge: T3 verdict with independent label noise

def _structural(prev, new, seg):
    for op in seg:
        out = op['out']
        if out is None: continue
        y = new.cols.get(out)
        if y is None or y.dtype.name != op['dtype'] or len(y) != len(new.ids) or len(y) == 0: return True
        if y.dtype.kind == 'f':
            if not np.isfinite(y).all(): return True
        elif y.min() < 0: return True
    np_, nn = len(prev.ids), len(new.ids)
    if nn > np_ or nn < .9 * np_: return True
    if nn > 1 and not (new.ids[1:] > new.ids[:-1]).all(): return True
    return False

def verify(tier, prev, new, seg, rng):
    if tier == 'T5':
        a = verify('T3', prev, new, seg, rng)
        u = rng.random()
        return (u > JUDGE_MISS) if a else (u < JUDGE_FALSE)
    if _structural(prev, new, seg): return True
    m = TIERS[tier]
    if m is None: return False
    n_prev = len(prev.ids)
    idx = np.arange(n_prev) if m == 'all' else np.sort(rng.choice(n_prev, size=min(m, n_prev), replace=False))
    produced = set(); need = set()
    for op in seg:
        for nm in (op.get('in1'), op.get('in2'), op.get('key')):
            if nm is not None and nm not in produced: need.add(nm)
        if op['out'] is not None: produced.add(op['out'])
    sub = State(prev.ids[idx], {c: prev.cols[c][idx] for c in need if c in prev.cols})
    for op in seg: sub = ref_op(sub, op)
    n_new = len(new.ids)
    if n_new == 0: return True
    pos = np.minimum(np.searchsorted(new.ids, sub.ids), n_new - 1)
    if not (new.ids[pos] == sub.ids).all(): return True
    gone = np.setdiff1d(prev.ids[idx], sub.ids, assume_unique=True)
    if len(gone):
        p2 = np.minimum(np.searchsorted(new.ids, gone), n_new - 1)
        if (new.ids[p2] == gone).any(): return True
    for c in produced:
        a = new.cols[c][pos]; b = sub.cols[c]
        if a.dtype.kind == 'f':
            if not np.allclose(a, b, rtol=1e-9, atol=1e-12): return True
        elif not np.array_equal(a, b): return True
    return False

# ---------------- execution harness -----------------
def run_segment(state, seg, rng, scale):
    """Execute steps; each step fails silently with probability P_INJ[kind]*scale."""
    u = rng.random(len(seg))
    for i, op in enumerate(seg):
        if u[i] < P_INJ[op['kind']] * scale:
            ft = FAULTS[int(rng.choice(len(FAULTS), p=FAULT_W))]
            state = faulty_op(state, op, ft, rng)
        else:
            state = ref_op(state, op)
    return state

def run_workflow(ops, data0, golden, cuts, tier, seed, scale, c0=0.0, c1=0.0, time_it=True):
    state = data0; pos = 0; units = 0.0; nsteps = 0; nver = 0; nalarm = 0; t_exec = 0.0; t_ver = 0.0
    vrng = np.random.default_rng([seed, 99])
    for end in cuts:
        seg = ops[pos:end]; L = end - pos; attempt = 0
        while True:
            frng = np.random.default_rng([seed, pos, attempt])
            t = perf_counter(); new = run_segment(state, seg, frng, scale); t_exec += perf_counter() - t
            units += L; nsteps += L
            if tier is None: break
            t = perf_counter(); alarm = verify(tier, state, new, seg, vrng); t_ver += perf_counter() - t
            units += c0 + c1 * L; nver += 1
            if not alarm: break
            nalarm += 1; attempt += 1
            if attempt > 400: break
        state = new; pos = end
    return dict(units=units, wall=t_exec + t_ver, ok=bool(states_equal(state, golden)), nver=nver, nalarm=nalarm, nsteps=nsteps)
