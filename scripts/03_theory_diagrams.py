#!/usr/bin/env python
"""The Hermite (Mehler / Wick) diagram rule for joint cumulants of functions of
jointly Gaussian variables -- statement, and three independent verifications.

THE RULE
========
Let ``t = (t_1..t_p)`` be jointly standard Gaussian with correlation matrix
``rho`` and let ``f_1..f_p`` be square-integrable, with Hermite coefficients

    a^u_k = E[f_u(t) He_k(t)],   t ~ N(0,1)     (so f_u = sum_k a^u_k He_k / k!)

Then

  MOMENTS   E[ prod_u f_u(t_u) ]
              = sum over ALL loopless multigraphs G on [p]
                  prod_u a^u_{deg_u(G)} * prod_{u<v} rho_uv^{e_uv} / e_uv!

  CUMULANTS kappa( f_1(t_1), ..., f_p(t_p) )
              = same sum restricted to CONNECTED G.

Proof.
  (i) Wick.  He_k(t) is the Wick power :t^k:, so
        E[ prod_u He_{k_u}(t_u) ] = sum over perfect matchings of the k_u legs
        with no leg matched inside its own vertex, each matched pair (u,v)
        contributing rho_uv.  Collecting matchings by their multigraph
        {e_uv} gives the multiplicity  prod_u k_u! / prod_{u<v} e_uv!  ,
        because vertex u splits its k_u legs into bundles
        (multinomial k_u!/prod_v e_uv!) and the two bundles of an edge are
        then matched in e_uv! ways.
  (ii) Substitute f_u = sum_k a^u_k He_k / k!.  The k_u! in the multiplicity
        cancels the 1/k_u! of the expansion and k_u becomes deg_u(G), giving
        the moment formula.
  (iii) Linked cluster.  The weight w(G) factorises over connected components
        (a^u_{deg} only sees u's own component), so
            sum_G w(G) = sum_{set partitions pi} prod_{B in pi} c(B),
            c(B) = sum over connected G on B of w(G).
        That is exactly the moment<->cumulant relation, so by uniqueness of
        Moebius inversion  kappa(f_1..f_p) = c([p]).                        []

REPEATED ARGUMENTS
==================
The layer map needs joint cumulants in which the same neuron appears several
times, e.g. kappa(x_i, x_i, x_k).  The formula above is a polynomial identity in
rho and therefore still valid at rho=1, but the series in the coincident edge
multiplicity then converges only algebraically.  Resumming those internal edges
turns a *block* of b coincident slots into a single vertex whose weight is

    beta^i_{b,d} := kappa( x_i, ..., x_i, He_d(t_i) )     (b copies of x_i)
                  = b! [u^b] ( E[e^{u x} He_d(t)] / E[e^{u x}] ),

so that, for blocks B_1..B_q at DISTINCT neurons i_1..i_q,

    kappa( x_{i_1}^(b_1), ..., x_{i_q}^(b_q) )
        = sum over connected loopless multigraphs G on [q]
             prod_b beta^{i_b}_{b_b, deg_b(G)} * prod rhohat^{e}/e!
          + Ursell (multi-cluster) corrections.

For total order r = sum b_b <= 4 the only Ursell corrections are

    (b) = (2,2):     - 2 C_{i1 i2}^2
    (b) = (2,1,1):   - 2 C_{i1 i2} C_{i1 i3}        (block 1 is the size-2 one)

with C = Cov(x_i, x_k).  They arise from the -Delta^2/2 term of
log(1+Delta) and are exactly what makes the whole thing vanish for linear f.

Run:  python scripts/03_theory_diagrams.py
"""
from __future__ import annotations

import itertools
import math
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, "/home/user/aicrowd")
from whestfloor.relu_moments import Phi, hermite_prob, phi  # noqa: E402

FAIL: list[str] = []


def report(name, got, want, tol, scale=1.0):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want)))) / scale
    ok = err <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<52s} err = {err:.3e}  (tol {tol:.1e})")
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# graph enumeration
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def connected_multigraphs(q: int, emax: int):
    """All connected loopless multigraphs on q labelled vertices with total
    edge multiplicity <= emax.  Returned as tuples of e_uv in pair order."""
    pairs = list(itertools.combinations(range(q), 2))
    out = []

    def rec(idx, left, acc):
        if idx == len(pairs):
            if _connected(q, pairs, acc):
                out.append(tuple(acc))
            return
        for e in range(left + 1):
            acc.append(e)
            rec(idx + 1, left - e, acc)
            acc.pop()

    if q == 1:
        return [()]
    rec(0, emax, [])
    return out


def _connected(q, pairs, e):
    par = list(range(q))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for (u, v), m in zip(pairs, e):
        if m:
            ru, rv = find(u), find(v)
            if ru != rv:
                par[ru] = rv
    return len({find(i) for i in range(q)}) == 1


def degrees(q, e):
    pairs = list(itertools.combinations(range(q), 2))
    d = [0] * q
    for (u, v), m in zip(pairs, e):
        d[u] += m
        d[v] += m
    return d


def diagram_cumulant(betas, rho, emax):
    """kappa over q blocks.  betas[b][d] is the vertex weight of block b at
    degree d (an array indexed by degree, length >= emax+1)."""
    q = len(betas)
    if q == 1:
        raise ValueError("single block handled by the exact single-site cumulant")
    pairs = list(itertools.combinations(range(q), 2))
    tot = 0.0
    for e in connected_multigraphs(q, emax):
        d = degrees(q, e)
        w = 1.0
        for b in range(q):
            w *= betas[b][d[b]]
        for (u, v), m in zip(pairs, e):
            w *= rho[u, v] ** m / math.factorial(m)
        tot += w
    return tot


def diagram_moment(a, rho, emax):
    """E[prod f_u] over p vertices: all loopless multigraphs (not just connected)."""
    p = len(a)
    pairs = list(itertools.combinations(range(p), 2))
    tot = 0.0

    def rec(idx, left, acc):
        nonlocal tot
        if idx == len(pairs):
            d = degrees(p, acc)
            w = 1.0
            for u in range(p):
                w *= a[u][d[u]]
            for (uu, vv), m in zip(pairs, acc):
                w *= rho[uu, vv] ** m / math.factorial(m)
            tot += w
            return
        for e in range(left + 1):
            acc.append(e)
            rec(idx + 1, left - e, acc)
            acc.pop()

    rec(0, emax, [])
    return tot


# ---------------------------------------------------------------------------
# generic cumulant from moments (for the MC reference)
# ---------------------------------------------------------------------------
def set_partitions(lst):
    if len(lst) == 1:
        yield [lst]
        return
    first, rest = lst[0], lst[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def joint_cumulant_mc(cols):
    """kappa(X_1..X_r) from samples; cols is a list of 1-D arrays (same length)."""
    r = len(cols)
    tot = 0.0
    for pi in set_partitions(list(range(r))):
        term = (-1) ** (len(pi) - 1) * math.factorial(len(pi) - 1)
        for B in pi:
            pr = np.ones_like(cols[0])
            for i in B:
                pr = pr * cols[i]
            term *= pr.mean()
        tot += term
    return tot


# ---------------------------------------------------------------------------
# ReLU Hermite / block coefficients
# ---------------------------------------------------------------------------
def relu_A(m, s, pmax, kmax):
    """A[p][k] = E[relu(m+st)^p He_k(t)] for p=0..pmax, k=0..kmax (scalars or arrays)."""
    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    al = m / s
    Pa, pa = Phi(al), phi(al)
    # truncated normal moments I_p = E[(t+al)^p 1{t>-al}]
    I = [Pa, al * Pa + pa]
    for p in range(2, pmax + 1):
        I.append(al * I[p - 1] + (p - 1) * I[p - 2])
    # a_k
    a = [m * Pa + s * pa, s * Pa]
    for k in range(2, kmax + 2):
        a.append(((-1.0) ** k) * s * hermite_prob(k - 2, al) * pa)
    A = [[np.zeros_like(al) for _ in range(kmax + 1)] for _ in range(pmax + 1)]
    for k in range(kmax + 1):
        A[0][k] = np.ones_like(al) if k == 0 else np.zeros_like(al)
    for p in range(1, pmax + 1):
        for k in range(kmax + 1):
            if k >= p - 1:
                A[p][k] = math.factorial(p) * s ** (p - 1) * a[k - p + 1]
            else:
                A[p][k] = math.factorial(p) / math.factorial(p - k) * s ** p * I[p - k]
    return A, a


def beta_block(A, b, kmax):
    """beta_{b,d} = b! [u^b] ( sum_p u^p A[p][d]/p!  /  sum_p u^p A[p][0]/p! )."""
    num = [A[p][0] * 0 for p in range(b + 1)]
    out = []
    den = [A[p][0] / math.factorial(p) for p in range(b + 1)]
    for d in range(kmax + 1):
        num = [A[p][d] / math.factorial(p) for p in range(b + 1)]
        # series division num/den
        c = [None] * (b + 1)
        for k in range(b + 1):
            acc = num[k]
            for j in range(1, k + 1):
                acc = acc - den[j] * c[k - j]
            c[k] = acc / den[0]
        out.append(c[b] * math.factorial(b))
    return out


def single_site_cumulants(A):
    """kappa_1..kappa_4 of x = relu(m+st) from raw moments A[p][0]."""
    mu = A[1][0]
    m2, m3, m4 = A[2][0], A[3][0], A[4][0]
    k2 = m2 - mu ** 2
    k3 = m3 - 3 * mu * m2 + 2 * mu ** 3
    k4 = m4 - 4 * mu * m3 - 3 * m2 ** 2 + 12 * mu ** 2 * m2 - 6 * mu ** 4
    return mu, k2, k3, k4


# ===========================================================================
# TEST 1 -- exact, polynomial f: diagram sum vs exact Gaussian expectation
# ===========================================================================
def gauss_hermite_prob(nq):
    k = np.arange(1, nq)
    J = np.diag(np.sqrt(k.astype(float)), -1) + np.diag(np.sqrt(k.astype(float)), 1)
    x, V = np.linalg.eigh(J)
    return x, V[0] ** 2


def test_polynomial(p=3, deg=3, nq=12, seed=1):
    """f_u polynomial of degree `deg` -> Hermite coefficients exact and finite,
    diagram sum finite.  Reference: tensor Gauss-Hermite (exact for polynomials)."""
    rng = np.random.default_rng(seed)
    L = np.tril(rng.standard_normal((p, p)))
    S = L @ L.T
    d = np.sqrt(np.diag(S))
    rho = S / np.outer(d, d)
    Lc = np.linalg.cholesky(rho + 1e-14 * np.eye(p))
    # f_u(t) = sum_k coef[u,k] He_k(t)/k!  -> a^u_k = coef[u,k]
    coef = rng.standard_normal((p, deg + 1))
    x, w = gauss_hermite_prob(nq)
    grid = np.array(np.meshgrid(*[x] * p, indexing="ij")).reshape(p, -1)
    wgt = np.ones(grid.shape[1])
    for u in range(p):
        wgt = wgt * w[np.array(np.meshgrid(*[np.arange(nq)] * p,
                                           indexing="ij")).reshape(p, -1)[u]]
    t = Lc @ grid                                    # correlated nodes
    F = np.array([sum(coef[u, k] * hermite_prob(k, t[u]) / math.factorial(k)
                      for k in range(deg + 1)) for u in range(p)])
    exact_mom = float(np.sum(wgt * np.prod(F, axis=0)))
    a = [list(coef[u]) + [0.0] * 8 for u in range(p)]
    diag_mom = diagram_moment(a, rho, emax=p * deg)
    report(f"p={p} moment: diagram sum == exact Gaussian integral", diag_mom, exact_mom, 1e-9)

    # cumulant: reference from all sub-moments via the partition formula
    def mom(subset):
        pr = np.ones(grid.shape[1])
        for u in subset:
            pr = pr * F[u]
        return float(np.sum(wgt * pr))

    ref = 0.0
    for pi in set_partitions(list(range(p))):
        term = (-1) ** (len(pi) - 1) * math.factorial(len(pi) - 1)
        for B in pi:
            term *= mom(B)
        ref += term
    got = diagram_cumulant([np.array(ai) for ai in a], rho, emax=p * deg)
    report(f"p={p} cumulant: connected diagrams == exact cumulant", got, ref, 1e-9)


# ===========================================================================
# TEST 2 -- ReLU joint cumulants (incl. repeated arguments) vs Monte Carlo
# ===========================================================================
def test_relu_joint(nsamp=40_000_000, emax=26, seed=3):
    print("\n(2) ReLU joint cumulants, repeated arguments included, vs Monte Carlo")
    rng = np.random.default_rng(seed)
    p = 4
    m = np.array([0.35, -0.40, 0.10, 0.80])
    s = np.array([1.00, 0.70, 1.30, 0.55])
    B = rng.standard_normal((p, p))
    S = B @ B.T + 0.6 * np.eye(p)
    dd = np.sqrt(np.diag(S))
    rho = S / np.outer(dd, dd)
    rhohat = rho - np.eye(p)
    print("    rho off-diagonal:", np.round(rhohat[np.triu_indices(p, 1)], 3))

    A, _ = relu_A(m, s, 4, emax)
    b1 = np.array([[A[1][k][u] for k in range(emax + 1)] for u in range(p)])
    b2 = np.array([[v[u] for v in beta_block(A, 2, emax)] for u in range(p)])
    b3 = np.array([[v[u] for v in beta_block(A, 3, emax)] for u in range(p)])
    # b_{*,0} must be 0 for a block to be attachable (isolated blocks excluded)
    for arr in (b1, b2, b3):
        arr[:, 0] = 0.0
    mu, k2x, k3x, k4x = single_site_cumulants(A)

    # covariance (for the Ursell terms), exact Mehler
    Cov = np.zeros((p, p))
    for u in range(p):
        for v in range(p):
            if u == v:
                Cov[u, v] = k2x[u]
            else:
                Cov[u, v] = sum(rhohat[u, v] ** k / math.factorial(k) * b1[u, k] * b1[v, k]
                                for k in range(1, emax + 1))

    # --- Monte Carlo ---
    L = np.linalg.cholesky(rho + 1e-12 * np.eye(p))
    acc = {}
    keys = [(0, 1), (0, 1, 2), (0, 0, 1), (0, 0, 0), (0, 1, 2, 3), (0, 0, 1, 2),
            (0, 0, 1, 1), (0, 0, 0, 1), (0, 0, 0, 0)]
    chunk = 2_000_000
    sums = {k: 0.0 for k in keys}
    # accumulate raw joint moments needed by the partition formula
    need = set()
    for k in keys:
        for r in range(1, len(k) + 1):
            for c in itertools.combinations(range(len(k)), r):
                need.add(tuple(sorted(k[i] for i in c)))
    raw = {t: 0.0 for t in need}
    done = 0
    while done < nsamp:
        nb = min(chunk, nsamp - done)
        g = rng.standard_normal((nb, p))
        t = g @ L.T
        x = np.maximum(m + s * t, 0.0)
        for tt in need:
            pr = np.ones(nb)
            for i in tt:
                pr *= x[:, i]
            raw[tt] += pr.sum()
        done += nb
    for tt in raw:
        raw[tt] /= done

    def kap_mc(key):
        r = len(key)
        tot = 0.0
        for pi in set_partitions(list(range(r))):
            term = (-1) ** (len(pi) - 1) * math.factorial(len(pi) - 1)
            for Bk in pi:
                term *= raw[tuple(sorted(key[i] for i in Bk))]
            tot += term
        return tot

    # --- diagram predictions ---
    def pred(key):
        blocks = {}
        for slot in key:
            blocks.setdefault(slot, 0)
            blocks[slot] += 1
        nodes = sorted(blocks)
        sizes = [blocks[u] for u in nodes]
        q = len(nodes)
        if q == 1:
            return {1: mu[nodes[0]], 2: k2x[nodes[0]], 3: k3x[nodes[0]],
                    4: k4x[nodes[0]]}[sizes[0]]
        table = {1: b1, 2: b2, 3: b3}
        betas = [table[sizes[b]][nodes[b]] for b in range(q)]
        sub = rhohat[np.ix_(nodes, nodes)]
        val = diagram_cumulant(betas, sub, emax)
        # Ursell corrections (total order 4 only)
        if sorted(sizes) == [2, 2]:
            val -= 2 * Cov[nodes[0], nodes[1]] ** 2
        if sorted(sizes) == [1, 1, 2]:
            h = sizes.index(2)
            others = [nodes[i] for i in range(3) if i != h]
            val -= 2 * Cov[nodes[h], others[0]] * Cov[nodes[h], others[1]]
        return val

    print(f"    N = {done:,}   emax = {emax}")
    print(f"    {'cumulant':<22s} {'diagram':>13s} {'monte carlo':>13s} {'diff':>11s} {'mc se':>10s}")
    scale = float(np.sqrt(np.mean(k2x)))
    for key in keys:
        pv, mv = pred(key), kap_mc(key)
        se = (len(key) ** 1.5) * scale ** len(key) / math.sqrt(done) * 3
        nm = "kappa(" + ",".join(f"x{i+1}" for i in key) + ")"
        flag = "ok " if abs(pv - mv) < 6 * se else "??"
        print(f"    [{flag}] {nm:<17s} {pv:13.6f} {mv:13.6f} {pv-mv:11.2e} {se:10.1e}")
        if abs(pv - mv) >= 6 * se:
            FAIL.append(nm)

    # Gaussian sanity: linear f must give zero for every r >= 3 cumulant
    print("    linear-f control (all r>=3 joint cumulants must vanish):")
    lin1 = np.zeros((p, emax + 1)); lin1[:, 1] = s
    lin2 = np.zeros((p, emax + 1)); lin2[:, 2] = 2 * s ** 2
    lin3 = np.zeros((p, emax + 1))                       # beta_{3,d} = 0 for linear f
    CovL = np.outer(s, s) * rho
    worst = 0.0
    for key in [(0, 1, 2), (0, 0, 1), (0, 1, 2, 3), (0, 0, 1, 2), (0, 0, 1, 1), (0, 0, 0, 1)]:
        blocks = {}
        for slot in key:
            blocks[slot] = blocks.get(slot, 0) + 1
        nodes = sorted(blocks); sizes = [blocks[u] for u in nodes]
        if len(nodes) == 1:
            continue
        table = {1: lin1, 2: lin2, 3: lin3}
        betas = [table[sizes[b]][nodes[b]] for b in range(len(nodes))]
        v = diagram_cumulant(betas, rhohat[np.ix_(nodes, nodes)], emax)
        if sorted(sizes) == [2, 2]:
            v -= 2 * CovL[nodes[0], nodes[1]] ** 2
        if sorted(sizes) == [1, 1, 2]:
            h = sizes.index(2)
            o = [nodes[i] for i in range(3) if i != h]
            v -= 2 * CovL[nodes[h], o[0]] * CovL[nodes[h], o[1]]
        worst = max(worst, abs(v))
    report("linear f: all r>=3 joint cumulants vanish", worst, 0.0, 1e-12)


if __name__ == "__main__":
    print("(1) exact check with polynomial f (machine precision expected)")
    test_polynomial(p=3, deg=3, nq=12, seed=1)
    test_polynomial(p=4, deg=2, nq=8, seed=2)
    test_relu_joint()
    print("\nFAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)
