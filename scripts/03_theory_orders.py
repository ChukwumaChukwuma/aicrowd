#!/usr/bin/env python
"""Layer cumulants kappa_3(z_j), kappa_4(z_j) for z_j = sum_i W_ij relu(z_i):
diagram enumeration, exact reference contraction, factorisation, order counting.

DECOMPOSITION
=============
    kappa_r(z_j) = sum_{i_1..i_r} W_{i_1 j}..W_{i_r j} kappa(x_{i_1},..,x_{i_r})
                 = sum over slot-partitions pi of [r]        (which slots share
                     a neuron; mult(pi) partitions of that type)
                   sum over connected multigraphs G on the blocks of pi
                     (1/prod e!) * INJ[ prod_B beta_{|B|,deg_B} * W^{|B|}
                                        * prod_edges rhohat^e ]
                 + Ursell corrections (r = 4 only, see 03_theory_diagrams.py)

``INJ`` is the index sum restricted to distinct neurons per block, evaluated as
   sum_sigma mu(sigma) * (unrestricted sum over the sigma-quotient graph),
   mu(sigma) = prod_S (-1)^{|S|-1}(|S|-1)!,
over set partitions sigma of the blocks in which no two merged blocks are
G-adjacent (rhohat has zero diagonal, so adjacent merges vanish identically).

COST CLASSES
============
A diagram whose quotient graph is a TREE factorises into
   matmul(n,n,n) -> elementwise products -> column sum,
costing (#tree edges) matmuls of 2n^3 plus O(n^2); and crucially the matmuls are
shared across diagrams (see FACTORED_PRIMITIVES below).  A quotient graph with
c independent cycles costs O(n^{3+c}) for all j at once and does NOT factorise.

Run:
  python scripts/03_theory_orders.py verify     # n=48, one layer, vs Monte Carlo
  python scripts/03_theory_orders.py orders     # per-diagram magnitudes, n=256
  python scripts/03_theory_orders.py cost       # FLOP model for the fast path
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/aicrowd")
_spec = importlib.util.spec_from_file_location(
    "thdiag", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_theory_diagrams.py"))
thdiag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(thdiag)  # type: ignore

connected_multigraphs = thdiag.connected_multigraphs
degrees = thdiag.degrees
relu_A = thdiag.relu_A
beta_block = thdiag.beta_block
single_site_cumulants = thdiag.single_site_cumulants
set_partitions = thdiag.set_partitions

FAIL: list[str] = []


# ---------------------------------------------------------------------------
# slot partitions
# ---------------------------------------------------------------------------
def _int_partitions(n, mx=None):
    mx = n if mx is None else mx
    if n == 0:
        yield []
        return
    for k in range(min(n, mx), 0, -1):
        for rest in _int_partitions(n - k, k):
            yield [k] + rest


def partition_types(r):
    out = []
    for sizes in _int_partitions(r):
        cnt = {}
        for s in sizes:
            cnt[s] = cnt.get(s, 0) + 1
        mult = math.factorial(r)
        for s in sizes:
            mult //= math.factorial(s)
        for c in cnt.values():
            mult //= math.factorial(c)
        out.append((tuple(sizes), mult))
    return out


# ---------------------------------------------------------------------------
# diagrams
# ---------------------------------------------------------------------------
class Diagram:
    __slots__ = ("sizes", "mult", "e", "q", "deg", "pairs", "order", "cycles", "tag")

    def __init__(self, sizes, mult, e):
        self.sizes, self.mult, self.e = sizes, mult, e
        self.q = len(sizes)
        self.pairs = list(itertools.combinations(range(self.q), 2))
        self.deg = degrees(self.q, e) if self.q > 1 else [0]
        self.order = sum(e)
        nedge = sum(1 for m in e if m)
        self.cycles = nedge - (self.q - 1)
        shape = {(1,): "single", (2,): "single", (3,): "single", (4,): "single"}.get(
            (self.q,), "")
        self.tag = f"{'+'.join(map(str, sizes)):<8s} e={'.'.join(map(str, e)) or '-':<10s}"


def catalogue(r, emax, max_cycles=99):
    """emax may be an int (total edge budget) or a dict {q: budget}."""
    out = []
    for sizes, mult in partition_types(r):
        q = len(sizes)
        if q == 1:
            out.append(Diagram(sizes, mult, ()))
            continue
        cap = emax[q] if isinstance(emax, dict) else emax
        for e in connected_multigraphs(q, cap):
            d = Diagram(sizes, mult, e)
            if d.cycles <= max_cycles:
                out.append(d)
    return out


# ---------------------------------------------------------------------------
# reference evaluation (all j at once)
# ---------------------------------------------------------------------------
def _quotient(dg, sigma):
    Q = len(sigma)
    where = {b: i for i, S in enumerate(sigma) for b in S}
    qe = np.zeros((Q, Q), dtype=int)
    for (u, v), m in zip(dg.pairs, dg.e):
        if m:
            qe[where[u], where[v]] += m
            qe[where[v], where[u]] += m
    return qe


def _leafmat(ctx, e, spec):
    """Rhat^e @ diag(prod of beta's) @ W^p  --  the ONLY matmul the tree
    diagrams ever need, and the reason the whole scheme fits in the budget.
    `spec` is a sorted tuple of (block_size, degree) pairs merged onto the node.
    Cached, so identical leaves across diagrams cost one matmul in total."""
    key = (e, spec)
    hit = ctx["cache"].get(key)
    if hit is not None:
        ctx["cache_hits"] += 1
        return hit
    n = ctx["n"]
    v = np.ones(n)
    p = 0
    for b, d in spec:
        v = v * ctx["beta"][b][d]
        p += b
    out = ctx["Ehat"][e] @ (v[:, None] * ctx["Wpow"][p])
    ctx["cache"][key] = out
    ctx["matmuls"] += 1
    return out


def _nodemat(ctx, spec):
    n = ctx["n"]
    v = np.ones(n)
    p = 0
    for b, d in spec:
        v = v * ctx["beta"][b][d]
        p += b
    return v[:, None] * ctx["Wpow"][p]


def _contract(dg, sigma, ctx):
    """Leaf-stripping contraction.  Every leaf removal is one cached matmul;
    a cyclic core falls back to einsum (O(n^{3+cycles}))."""
    Q = len(sigma)
    specs = [tuple(sorted((dg.sizes[b], dg.deg[b]) for b in S)) for S in sigma]
    qe = _quotient(dg, sigma)
    alive = list(range(Q))
    X = {}                                        # accumulated j-dependent factors
    while len(alive) > 1:
        leaf = None
        for i in alive:
            if sum(1 for k in alive if k != i and qe[i, k]) == 1:
                leaf = i
                break
        if leaf is None:
            break
        k = next(kk for kk in alive if kk != leaf and qe[leaf, kk])
        e = qe[leaf, k]
        blk = X.pop(leaf, None)
        if blk is None:
            mat = _leafmat(ctx, e, specs[leaf])    # cached, shared
        else:
            mat = ctx["Ehat"][e] @ (_nodemat(ctx, specs[leaf]) * blk)
            ctx["matmuls"] += 1
        X[k] = mat if k not in X else X[k] * mat
        alive.remove(leaf)
    if len(alive) == 1:
        i = alive[0]
        out = _nodemat(ctx, specs[i])
        if i in X:
            out = out * X[i]
        return out.sum(axis=0)
    # cyclic core.  With max_cycles = 1 and every remaining node of degree >= 2
    # the core is a simple cycle v_0 - v_1 - ... - v_{Q-1} - v_0, and
    #     T_z = tr( prod_i diag(node_i[:,z]) Ehat^{e(i,i+1)} ),
    # which costs (Q-2) matmuls per column z, i.e. O(n^4) for Q = 3.
    ctx["cyclic"] += 1
    mats = {}
    for i in alive:
        m = _nodemat(ctx, specs[i])
        if i in X:
            m = m * X[i]
        mats[i] = m
    cyc = [alive[0]]
    while len(cyc) < len(alive):
        prev = cyc[-1]
        nxt = next(k for k in alive if k not in cyc and qe[prev, k])
        cyc.append(nxt)
    Q = len(cyc)
    n = ctx["n"]
    Es = [ctx["Ehat"][qe[cyc[i], cyc[(i + 1) % Q]]] for i in range(Q)]
    out = np.empty(n)
    blk = max(1, 2 ** 22 // (n * n))
    for lo in range(0, n, blk):
        hi = min(n, lo + blk)
        acc = None
        for i in range(1, Q):
            Di = mats[cyc[i]][:, lo:hi].T[:, :, None] * Es[i]      # (b,n,n)
            acc = Di if acc is None else acc @ Di
        head = mats[cyc[0]][:, lo:hi].T[:, :, None] * Es[0]
        out[lo:hi] = np.einsum("bij,bji->b", head, acc)
    return out


def eval_diagram(dg, ctx, exact_inj=True, max_cycles=99):
    n = ctx["n"]
    tot = np.zeros(n)
    sigmas = list(set_partitions(list(range(dg.q)))) if exact_inj else \
        [[[b] for b in range(dg.q)]]
    for sigma in sigmas:
        bad = False
        for S in sigma:
            for u, v in itertools.combinations(S, 2):
                if dg.e[dg.pairs.index((min(u, v), max(u, v)))] > 0:
                    bad = True
        if bad:
            continue
        qe = _quotient(dg, sigma)
        Q = len(sigma)
        nedge = int(sum(1 for i in range(Q) for k in range(i + 1, Q) if qe[i, k]))
        if nedge - (Q - 1) > max_cycles:
            continue
        w = 1.0
        for S in sigma:
            w *= (-1) ** (len(S) - 1) * math.factorial(len(S) - 1)
        tot += w * _contract(dg, sigma, ctx)
    coef = dg.mult / math.prod(math.factorial(m) for m in dg.e)
    return coef * tot


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
def _cap(emax):
    return max(emax.values()) if isinstance(emax, dict) else emax


def build_ctx(m, s, R, W, emax):
    n = W.shape[0]
    emax = _cap(emax)
    A, _ = relu_A(m, s, 4, emax)
    mu, k2, k3, k4 = single_site_cumulants(A)
    beta = {}
    for b in (1, 2, 3):
        bb = np.array(beta_block(A, b, emax))
        bb[0] = 0.0
        beta[b] = bb
    beta[4] = np.zeros((emax + 1, n))
    for b, kv in ((1, mu), (2, k2), (3, k3), (4, k4)):
        beta[b][0] = kv                       # degree 0 <-> isolated block
    Rhat = R - np.eye(n)
    Ehat = [np.ones((n, n)), Rhat.copy()]
    for e in range(2, emax + 1):
        Ehat.append(Ehat[-1] * Rhat)
    C = np.zeros((n, n))
    for e in range(1, emax + 1):
        C += (Ehat[e] / math.factorial(e)) * np.outer(beta[1][e], beta[1][e])
    np.fill_diagonal(C, k2)
    return dict(n=n, W=W, Wpow={p: W ** p for p in range(1, 5)}, Ehat=Ehat,
                beta=beta, C=C, mu=mu, k2=k2, k3=k3, k4=k4, emax=emax, A=A, m=m, s=s,
                cache={}, cache_hits=0, matmuls=0, cyclic=0)


def ursell(ctx, exact_inj=True):
    W, C = ctx["W"], ctx["C"]
    Ch = C - np.diag(np.diag(C))
    W2 = ctx["Wpow"][2]
    out = {}
    out["2+2      ursell    "] = -6.0 * np.einsum("iz,iz->z", W2, (Ch * Ch) @ W2)
    t = np.einsum("iz,iz->z", W2, (Ch @ W) ** 2)
    if exact_inj:
        t = t - np.einsum("iz,iz->z", W2, (Ch * Ch) @ W2)
    out["2+1+1    ursell    "] = -12.0 * t
    return out


def layer_cumulants(m, s, R, W, emax, exact_inj=True, max_cycles=99, rs=(3, 4)):
    ctx = build_ctx(m, s, R, W, emax)
    per = {}
    for r in rs:
        for dg in catalogue(r, emax, max_cycles):
            per[f"r{r} {dg.tag}"] = eval_diagram(dg, ctx, exact_inj, max_cycles)
    if 4 in rs:
        for k, v in ursell(ctx, exact_inj).items():
            per[f"r4 {k}"] = v
    return per, ctx


def totals(per):
    k3 = sum(v for k, v in per.items() if k.startswith("r3"))
    k4 = sum(v for k, v in per.items() if k.startswith("r4"))
    return k3, k4


# ===========================================================================
# VERIFY: one layer, exactly-Gaussian input, against Monte Carlo
# ===========================================================================
def make_case(n, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n)) / math.sqrt(n)
    S = B @ B.T + 0.35 * np.eye(n)             # a realistic-looking correlation
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    s = 0.7 + 0.5 * rng.random(n)
    m = 0.9 * s * rng.standard_normal(n)
    W = rng.standard_normal((n, n)) * math.sqrt(2.0 / n)
    return m, s, R, W


EMAX = {2: 12, 3: 8, 4: 6}


def verify(n=48, nsamp=120_000_000, emax=EMAX, seed=11):
    print(f"(verify) n={n}  N={nsamp:,}  emax={emax}")
    m, s, R, W = make_case(n, seed)
    off = R[~np.eye(n, dtype=bool)]
    print(f"    |rho| mean {np.abs(off).mean():.3f}  rms {np.sqrt((off**2).mean()):.3f}"
          f"  max {np.abs(off).max():.3f}   alpha rms {np.sqrt(((m/s)**2).mean()):.2f}")
    t0 = time.time()
    per, ctx = layer_cumulants(m, s, R, W, emax, max_cycles=1)
    k3p, k4p = totals(per)
    print(f"    diagram sum: {time.time()-t0:.1f}s   {len(per)} diagrams, "
          f"{ctx['matmuls']} distinct matmuls, {ctx['cache_hits']} cache hits, "
          f"{ctx['cyclic']} cyclic contractions")

    L = np.linalg.cholesky(R + 1e-12 * np.eye(n))
    rng = np.random.default_rng(seed + 1)
    S1 = np.zeros(n); S2 = np.zeros(n); S3 = np.zeros(n); S4 = np.zeros(n)
    XS = np.zeros(n)
    chunk, done = 200_000, 0
    t0 = time.time()
    while done < nsamp:
        nb = min(chunk, nsamp - done)
        t = rng.standard_normal((nb, n)) @ L.T
        x = np.maximum(m + s * t, 0.0)
        XS += x.sum(0)
        z = x @ W
        S1 += z.sum(0); z2 = z * z
        S2 += z2.sum(0); S3 += (z2 * z).sum(0); S4 += (z2 * z2).sum(0)
        done += nb
    S1 /= done; S2 /= done; S3 /= done; S4 /= done; XS /= done
    var = S2 - S1 ** 2
    k3m = S3 - 3 * S1 * S2 + 2 * S1 ** 3
    k4m = S4 - 4 * S1 * S3 - 3 * S2 ** 2 + 12 * S1 ** 2 * S2 - 6 * S1 ** 4
    sd = np.sqrt(var)
    se3 = math.sqrt(6.0 / done) * (sd ** 3).mean()
    se4 = math.sqrt(24.0 / done) * (sd ** 4).mean()
    print(f"    monte carlo: {time.time()-t0:.0f}s")

    # mean / covariance predicted from the same machinery (Gaussian input -> exact)
    mu_p = ctx["mu"] @ W
    print(f"    mean of z:  rms err {np.sqrt(((mu_p-S1)**2).mean()):.2e} "
          f"(mc se {np.sqrt((var/done).mean()):.1e})")
    Cp = W.T @ ctx["C"] @ W
    print(f"    var  of z:  rms err {np.sqrt(((np.diag(Cp)-var)**2).mean()):.2e}")
    for nm, p, mm, se in (("kappa_3", k3p, k3m, se3), ("kappa_4", k4p, k4m, se4)):
        e = p - mm
        ok = np.sqrt((e ** 2).mean()) < 4 * se
        print(f"    [{'ok ' if ok else 'FAIL'}] {nm}: rms |diagram - mc| = "
              f"{np.sqrt((e**2).mean()):.3e}   mc se {se:.1e}   "
              f"rms {nm} = {np.sqrt((mm**2).mean()):.3e}")
        if not ok:
            FAIL.append(nm)

    # what the truncations cost
    print("    truncation study (rms error against the Monte Carlo, "
          f"mc se {se3:.1e}/{se4:.1e}):")
    for label, em, kw in (
            ("full (reference)          ", emax, dict(max_cycles=1)),
            ("tree diagrams only        ", emax, dict(max_cycles=0)),
            ("no injectivity correction ", emax, dict(max_cycles=1, exact_inj=False)),
            ("emax 6/4/3                ", {2: 6, 3: 4, 4: 3}, dict(max_cycles=1)),
            ("emax 4/3/2                ", {2: 4, 3: 3, 4: 2}, dict(max_cycles=1)),
            ("emax 3/2/2                ", {2: 3, 3: 2, 4: 2}, dict(max_cycles=1)),
            ("emax 2/2/2 trees only     ", {2: 2, 3: 2, 4: 2}, dict(max_cycles=0)),
            ("coincident blocks only    ", {2: 12, 3: 0, 4: 0}, dict(max_cycles=1)),
    ):
        p2, _ = layer_cumulants(m, s, R, W, em, **kw)
        a3, a4 = totals(p2)
        print(f"      {label} k3 {np.sqrt(((a3-k3m)**2).mean()):.3e}   "
              f"k4 {np.sqrt(((a4-k4m)**2).mean()):.3e}")


# ===========================================================================
# ORDERS: per-diagram magnitudes on the real n=256 network
# ===========================================================================
def real_layer_stats(n=256, depth=32, seed=0, kmax=16):
    from whestfloor.mc import make_mlp
    from whestfloor.relu_moments import relu_cov_mehler, relu_mean
    W = [w.astype(np.float64) for w in make_mlp(n, depth, seed=seed)]
    mu, C = np.zeros(n), np.eye(n)
    out = []
    for w in W:
        m = w.T @ mu
        S = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diag(S), 1e-12))
        out.append((m, s, S / np.outer(s, s), w))
        C = relu_cov_mehler(S, m, s, kmax=kmax)
        mu = relu_mean(m, s)
    return out


def orders(layers=(1, 7, 15, 31), emax=8, n=256):
    stats = real_layer_stats(n=n)
    for li in layers:
        m, s, R, W = stats[li]
        if li + 1 < len(stats):
            Wn = stats[li + 1][3]
        else:
            Wn = W
        per, ctx = layer_cumulants(m, s, R, Wn, emax, max_cycles=1)
        k3, k4 = totals(per)
        sd_next = np.sqrt(np.diag(Wn.T @ ctx["C"] @ Wn))
        print(f"\n=== layer {li+1} -> {li+2}   |rho|mean={np.abs(R[~np.eye(n,dtype=bool)]).mean():.3f} "
              f" sigma_next mean={sd_next.mean():.3f}")
        print(f"    total kappa_3 rms {np.sqrt((k3**2).mean()):.4e}  "
              f"gamma_3 rms {np.sqrt(((k3/sd_next**3)**2).mean()):.4f}")
        print(f"    total kappa_4 rms {np.sqrt((k4**2).mean()):.4e}  "
              f"gamma_4 rms {np.sqrt(((k4/sd_next**4)**2).mean()):.4f}")
        items = sorted(per.items(), key=lambda kv: -np.sqrt((kv[1] ** 2).mean()))
        print(f"    {'diagram':<32s} {'rms':>10s} {'frac':>7s}  cycles")
        tot3 = np.sqrt((k3 ** 2).mean()); tot4 = np.sqrt((k4 ** 2).mean())
        shown = 0
        for k, v in items:
            rms = float(np.sqrt((v ** 2).mean()))
            den = tot3 if k.startswith("r3") else tot4
            if rms / den < 2e-3 or shown > 26:
                continue
            shown += 1
            print(f"    {k:<32s} {rms:10.3e} {rms/den:7.3f}")


# ===========================================================================
# COST: FLOP model of the factorised fast path
# ===========================================================================
def cost(n=256, K=6):
    """matmul (2n^3) counts of the recommended factorised implementation."""
    mm = 2 * n ** 3
    rows = [
        ("Sigma = W^T C W", 2, "exact, always needed"),
        ("M_e = Rhat^e diag(a_e) W", K, "tree leaves, size-1 blocks; shared by r=3 and r=4"),
        ("P_e = Rhat^e diag(c_e) W^2", K, "tree leaves, size-2 blocks (r=4 only)"),
        ("Y_D = Rhat^D diag(eta_D) W^2", K, "injectivity correction, 2 merged size-1 leaves"),
        ("Z_T = Rhat^T diag(theta_T) W^3", K, "injectivity correction, 3 merged leaves (r=4)"),
        ("U_q -> Rhat^q U_q (4-vertex path)", K, "r=4 path chain, one per middle multiplicity"),
        ("Chat^2 W^2  and  Chat W", 2, "Ursell terms"),
        ("Cov(x) Mehler series", 0, "O(K n^2), not a matmul"),
    ]
    tot = 0
    print(f"n={n}  K={K}   one matmul unit = 2n^3 = {mm:.3e} FLOPs")
    print(f"    {'primitive':<40s} {'#':>3s} {'FLOPs':>11s}")
    for nm, c, note in rows:
        tot += c
        print(f"    {nm:<40s} {c:3d} {c*mm:11.3e}   {note}")
    n2 = 60 * K * n * n
    print(f"    {'elementwise / column sums (~60 K n^2)':<40s} {'':>3s} {n2:11.3e}")
    print(f"    TOTAL per layer  {tot} matmuls + O(K n^2) = {tot*mm + n2:.3e} FLOPs")
    print(f"    32 layers: {32*(tot*mm+n2):.3e} FLOPs   (free-compute ceiling 2.72e10)")
    print(f"    one non-factorising 3-cycle, all j, exact: 2 n^4 = {2*n**4:.2e} FLOPs "
          f"= {2*n**4/mm:.0f} matmul units")
    print(f"    same 3-cycle via rank-k truncation of Rhat: ~2 k^2 n^2 + 2 k^3 n ; "
          f"k=32 -> {2*32**2*n*n + 2*32**3*n:.2e} FLOPs = "
          f"{(2*32**2*n*n+2*32**3*n)/mm:.1f} matmul units")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if what == "verify":
        verify()
    elif what == "orders":
        orders()
    elif what == "cost":
        cost()
    else:
        verify(); orders(); cost()
    print("\nFAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)
