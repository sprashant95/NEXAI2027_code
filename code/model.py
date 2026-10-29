"""Analytic model for verified checkpointing of agentic workflows.
Step cost = 1 unit. A verification after a segment of L steps costs c0 + c1*L units.
Verifier is local: it judges only the segment it just saw.
  recall r   : P(alarm | segment contains a silent error)
  false alarm f : P(alarm | segment is clean)
On alarm the state is restored to the last accepted checkpoint and the segment is re-run.
"""
import numpy as np

def r_eff(r, f):
    return (r - f) / (1.0 - f)

def c_eff(c0, c1):
    return c0 / (1.0 + c1)

def seg_pass_prob(q, r, f):
    """P(segment attempt is accepted) with q = P(segment has no silent error)."""
    return q * (1 - f) + (1 - q) * (1 - r)

def seg_clean_given_pass(q, r, f):
    return q * (1 - f) / seg_pass_prob(q, r, f)

def plan_cost(seg_q, seg_len, c0, c1, r, f):
    """Exact expected cost and success probability of a schedule given segment clean-probs/lengths."""
    seg_q = np.asarray(seg_q, float); seg_len = np.asarray(seg_len, float)
    rho = seg_pass_prob(seg_q, r, f)
    cost = np.sum((seg_len * (1 + c1) + c0) / rho)
    succ = np.prod(seg_q * (1 - f) / rho)
    return float(cost), float(succ)

def uniform_cost(n, k, p, c0, c1, r, f):
    """Homogeneous p, segments of length k (last one shorter)."""
    k = int(k)
    full, rem = divmod(n, k)
    lens = [k] * full + ([rem] if rem else [])
    q = [(1 - p) ** L for L in lens]
    return plan_cost(q, lens, c0, c1, r, f)

def law_k(p, c0, c1, r, f):
    """Closed-form interval, Eq. (k*): sqrt(c^2 + c/(r_eff p)) - c."""
    re = r_eff(r, f)
    if re <= 0:
        return np.inf
    c = c_eff(c0, c1)
    return float(np.sqrt(c * c + c / (re * p)) - c)

def overhead_law(p, c0, c1, r, f):
    """First-order relative overhead of the optimal schedule over bare execution."""
    re = r_eff(r, f)
    c = c_eff(c0, c1)
    return (1 + c1) / (1 - f) * (1 + 2 * np.sqrt(max(re, 0) * p * c)) - 1

def best_uniform(n, p, c0, c1, r, f):
    best = None
    for k in range(1, n + 1):
        cst, s = uniform_cost(n, k, p, c0, c1, r, f)
        if best is None or cst < best[1]:
            best = (k, cst, s)
    return best

def dp_schedule(p_steps, c0, c1, r, f):
    """Exact cost-optimal checkpoint placement for per-step error probabilities p_steps."""
    p_steps = np.asarray(p_steps, float)
    n = len(p_steps)
    logq = np.concatenate([[0.0], np.cumsum(np.log1p(-p_steps))])
    D = np.full(n + 1, np.inf); D[0] = 0.0; prev = np.zeros(n + 1, int)
    for j in range(1, n + 1):
        i = np.arange(0, j)
        L = j - i
        q = np.exp(logq[j] - logq[i])
        rho = seg_pass_prob(q, r, f)
        cand = D[i] + (L * (1 + c1) + c0) / rho
        a = int(np.argmin(cand)); D[j] = cand[a]; prev[j] = i[a]
    cuts = []; j = n
    while j > 0:
        cuts.append(j); j = prev[j]
    cuts = sorted(cuts)
    lens = np.diff([0] + cuts)
    return cuts, float(D[n])

def schedule_stats(p_steps, cuts, c0, c1, r, f):
    p_steps = np.asarray(p_steps, float)
    edges = [0] + list(cuts)
    lens = np.diff(edges)
    q = [np.prod(1 - p_steps[edges[i]:edges[i + 1]]) for i in range(len(lens))]
    return plan_cost(q, lens, c0, c1, r, f)
