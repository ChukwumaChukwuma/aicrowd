#!/usr/bin/env python
"""Is the top of the leaderboard a Gaussian mixture?  Measure the ceiling first.

Telemetry (public per-MLP pages, no auth) puts the top two entries at 6.30e9 and
2.43e10 billed FLOPs with raw MSE 9.0e-9 and 3.6e-9.  `docs/cost_floor.md` rules
out a sampler at that price and `docs/traj_closure.md` rules out a moment
closure at that accuracy, so what is left is a deterministic method costing a
handful of covariance propagations.  A `K`-component Gaussian mixture is exactly
that shape: `K` propagations, each component rectified in closed form.

**Every mixture method's answer has the same form**

    E[relu(z^L_j)]  ~=  sum_k w_k relu_mean(m_kj, s_kj),

so before building a propagator this script measures the best that form can
possibly do: take the *exact* law of `z^32` from Monte Carlo, cut it into `K`
cells along a chosen `r`-dimensional frame, and evaluate the formula with the
*true* per-cell conditional mean and standard deviation.  No propagation error,
no closure error, no fitting — only the representation error of a `K`-component
mixture.  That is a ceiling on the whole family, and it is cheap.

Modes
-----
``ceiling``   the ``(r, K)`` ceiling table, three frames, unbiased error.
``anatomy``   where the closure error lives in the eigenbasis, and the
              per-direction non-Gaussianity — the two premises the mixture
              hypothesis rests on, measured rather than assumed.
``propagate`` the self-contained mixture propagator: raw MSE and billed ``F``
              in a real ``flopscope.BudgetContext``.

Errors are reported as ``mean_i (p_i - a_i)(p_i - b_i)`` against two independent
reference halves, which removes the reference's own Monte-Carlo variance
exactly (``whestfloor/harness.py``).  The cell statistics come from a *third*
independent stream, so the predictor is independent of both halves and the
estimate is unbiased for the finite-sample predictor's true MSE; the run is
repeated at half the cell-stream size so the predictor's own noise is visible
rather than assumed away.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import forward_pass_flops  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.mixture import (  # noqa: E402
    CellAccumulator,
    MixtureState,
    frame_eig,
    frame_input,
    frame_kurt,
    gauss_hermite,
    lloyd,
    pooled_moments,
    predict_state,
)
from whestfloor.relu_moments import relu_mean  # noqa: E402
from whestfloor.trajclosure import run as chain_run  # noqa: E402

ART = Path(os.environ.get("WHEST_ARTIFACTS", "/tmp/whest")) / "mixture"

#: Seeds disjoint from every other page in this repository (docs/traj_closure.md
#: uses 960000+, the official suite uses the whestbench protocol).
MLP_SEED_BASE = 970_000


# ---------------------------------------------------------------------------
def stream_z(W, n_samples, seed, chunk=8192, want_x=False):
    """Yield ``(X, Z)`` chunks of input and final-layer *pre-activation*."""
    n = W[0].shape[0]
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        h = x
        for li, w in enumerate(W):
            z = h @ w
            if li == len(W) - 1:
                break
            h = np.maximum(z, np.float32(0.0))
        yield (x if want_x else None), z
        done += nb


def collect(W, n_samples, seed, chunk=8192, want_x=False):
    n = W[0].shape[0]
    Z = np.empty((n_samples, n), dtype=np.float32)
    X = np.empty((n_samples, n), dtype=np.float32) if want_x else None
    off = 0
    for x, z in stream_z(W, n_samples, seed, chunk, want_x):
        nb = z.shape[0]
        Z[off:off + nb] = z
        if want_x:
            X[off:off + nb] = x
        off += nb
    return X, Z


def truth_mean(W, n_samples, seed, chunk=8192):
    n = W[0].shape[0]
    acc = np.zeros(n, dtype=np.float64)
    tot = 0
    for _, z in stream_z(W, n_samples, seed, chunk):
        acc += np.maximum(z, 0.0).sum(axis=0, dtype=np.float64)
        tot += z.shape[0]
    return acc / tot


def umse(p, a, b):
    """Unbiased true MSE with the reference's own sampling variance removed."""
    return float(np.mean((p - a) * (p - b)))


# ---------------------------------------------------------------------------
#: (frame, r, [K, ...]).  ``r = 0`` is the K = 1 Gaussian oracle.
PLAN_FULL = [
    ("eig", 1, [2, 3, 4, 6, 8, 12, 16, 32, 64, 256, 1024]),
    ("eig", 2, [4, 6, 8, 16, 64, 256]),
    ("eig", 4, [6, 16, 64, 256]),
    ("eig", 8, [6, 64, 256]),
    ("eig", 16, [6, 64, 256]),
    ("eig", 32, [6, 64, 256]),
    ("kurt", 1, [6, 64]), ("kurt", 2, [6, 64]), ("kurt", 4, [6, 64]),
    ("input", 1, [6, 64]), ("input", 2, [6, 64]), ("input", 4, [6, 64]),
]
PLAN_QUICK = [("eig", 1, [6, 64]), ("eig", 4, [6, 64]), ("input", 1, [6])]


def _build_specs(plan, Zp, Xp):
    """Frames and Lloyd centroids from the pilot.  Returns specs + m0."""
    m0, U, ev = frame_eig(Zp)
    Zc = Zp.astype(np.float64) - m0
    rmax_k = max([r for (f, r, _) in plan if f == "kurt"], default=0)
    rmax_x = max([r for (f, r, _) in plan if f == "input"], default=0)
    Uk = frame_kurt(Zp, rmax_k) if rmax_k else None
    Ux = frame_input(Xp, Zp, rmax_x) if rmax_x else None
    Tp = {"eig": Zc @ U, "kurt": (Zc @ Uk) if Uk is not None else None,
          "input": (Xp.astype(np.float64) @ Ux) if Ux is not None else None}
    Uf = {"eig": U / np.sqrt(ev), "kurt": Uk, "input": Ux}
    Tp["eig"] = Tp["eig"] / np.sqrt(ev)
    specs = []
    for fname, r, Ks in plan:
        for K in Ks:
            C = lloyd(np.ascontiguousarray(Tp[fname][:, :r]), K, seed=1234 + K)
            specs.append((fname, r, K, "x" if fname == "input" else "z",
                          np.ascontiguousarray(Uf[fname][:, :r],
                                               dtype=np.float32), C))
    return specs, m0, ev


def _run_stream(W, specs, m0, n_samples, seed, chunk, marks):
    """One independent stream.  Returns ``{size: (preds, truth)}``.

    ``preds[c]`` is configuration ``c``'s answer and ``truth`` the plain sample
    mean of ``relu(z^32)`` **from the very same samples**.  Pairing them is the
    whole point: both are averages over one empirical distribution, so their
    difference cancels the leading Monte-Carlo fluctuation, and the product of
    two such differences from independent streams is an unbiased estimate of
    the squared model error with a variance far below what an independent
    reference could give at any affordable sample count.
    """
    n = 256
    accs = [CellAccumulator(C, n) for (*_, C) in specs]
    tsum = np.zeros(n)
    done = 0
    out = {}
    nxt = sorted(marks)
    m0f = np.asarray(m0, dtype=np.float32)
    for x, z in stream_z(W, n_samples, seed, chunk, want_x=True):
        B = z.shape[0]
        ZZ = np.empty((B, 2 * n), dtype=np.float32)
        ZZ[:, :n] = z
        np.multiply(z, z, out=ZZ[:, n:])
        tsum += np.maximum(z, np.float32(0.0)).sum(axis=0, dtype=np.float64)
        zc = z - m0f
        for acc, (_, _, _, kind, U, _) in zip(accs, specs):
            acc.add(ZZ, (zc if kind == "z" else x) @ U)
        done += B
        while nxt and done >= nxt[0]:
            sz = nxt.pop(0)
            out[sz] = ([predict_state(*a.state(), n) for a in accs],
                       tsum / done, pooled_moments(*accs[0].state(), n), done)
    return out


def mode_ceiling(args) -> int:
    ART.mkdir(parents=True, exist_ok=True)
    plan = PLAN_QUICK if args.quick else PLAN_FULL
    N = args.n_cells
    marks = [N // 4, N // 2, N]
    acc_rows: dict[tuple, dict[int, list]] = {}
    base: dict[str, dict[int, list]] = {"gauss": {}, "chain": {}, "pt": {}}

    print("# mixture ceiling: the best that  sum_k w_k relu_mean(m_k, s_k)  can do")
    print(f"# {args.mlps} MLPs, two independent streams of {N:,} samples each,")
    print("# paired same-stream residuals, Richardson-extrapolated in 1/N")
    print()

    for i in range(args.mlps):
        t0 = time.time()
        seed = MLP_SEED_BASE + i
        W = make_mlp(256, 32, seed)
        Xp, Zp = collect(W, args.n_pilot, seed=5_000_000 + i, want_x=True)
        specs, m0, ev = _build_specs(plan, Zp, Xp)
        del Xp, Zp
        t1 = time.time()
        s1 = _run_stream(W, specs, m0, N, 6_000_000 + 2 * i, args.chunk, marks)
        s2 = _run_stream(W, specs, m0, N, 6_000_000 + 2 * i + 1, args.chunk, marks)
        mu_chain = chain_run(W)[-1]
        for sz in marks:
            p1, t_1, (m1, v1), _ = s1[sz]
            p2, t_2, (m2, v2), _ = s2[sz]
            base["gauss"].setdefault(sz, []).append(float(np.mean(
                (relu_mean(m1, np.sqrt(v1)) - t_1)
                * (relu_mean(m2, np.sqrt(v2)) - t_2))))
            base["pt"].setdefault(sz, []).append(float(np.mean(
                (np.maximum(m1, 0.0) - t_1) * (np.maximum(m2, 0.0) - t_2))))
            base["chain"].setdefault(sz, []).append(
                float(np.mean((mu_chain - t_1) * (mu_chain - t_2))))
            for c, sp in enumerate(specs):
                acc_rows.setdefault(sp[:3], {}).setdefault(sz, []).append(
                    float(np.mean((p1[c] - t_1) * (p2[c] - t_2))))
        print(f"  mlp {i + 1}/{args.mlps} seed={seed}  pilot {t1 - t0:.0f}s "
              f"streams {time.time() - t1:.0f}s   K=1 oracle "
              f"{base['gauss'][N][-1]:.4e}  chain {base['chain'][N][-1]:.4e}",
              flush=True)
        # partial dump after every MLP: a 40-minute sweep should never be
        # all-or-nothing.
        (ART / f"ceiling{args.tag}_partial.json").write_text(json.dumps(
            {"done": i + 1,
             "table": {"|".join(map(str, k)):
                       {str(sz): list(map(float, v)) for sz, v in dd.items()}
                       for k, dd in acc_rows.items()},
             "baselines": {k: {str(sz): list(map(float, v))
                               for sz, v in dd.items()}
                           for k, dd in base.items()}}, indent=1))

    def rich(d):
        """2 * MSE(N) - MSE(N/2): the 1/N term removed."""
        return 2.0 * float(np.mean(d[N])) - float(np.mean(d[N // 2]))

    gb = rich(base["gauss"])
    out = {"n_mlps": args.mlps, "n_cells": N, "n_pilot": args.n_pilot,
           "mlp_seeds": [MLP_SEED_BASE + i for i in range(args.mlps)],
           "baselines": {k: {str(s): float(np.mean(v)) for s, v in d.items()}
                         for k, d in base.items()},
           "baselines_extrap": {k: rich(d) for k, d in base.items()},
           "table": {"|".join(map(str, k)):
                     {str(s): float(np.mean(v)) for s, v in d.items()}
                     for k, d in acc_rows.items()},
           "table_extrap": {"|".join(map(str, k)): rich(d)
                            for k, d in acc_rows.items()}}

    print()
    print(f"# baselines, raw MSE (extrapolated), {args.mlps} MLPs")
    print(f"    analytic Gaussian chain, no oracle      "
          f"{rich(base['chain']):.4e}")
    print(f"    K=1 Gaussian, EXACT mean and variance   {gb:.4e}")
    print(f"    relu(exact mean), i.e. s -> 0           {rich(base['pt']):.4e}")
    print()
    hdr = (f"  {'frame':>6} {'r':>3} {'K':>5}   {'N/4':>11} {'N/2':>11} "
           f"{'N':>11}   {'extrap':>11} {'x K=1':>7} {'rms':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(acc_rows, key=lambda t: (t[0], t[1], t[2])):
        d = acc_rows[key]
        e = rich(d)
        print(f"  {key[0]:>6} {key[1]:>3} {key[2]:>5}   "
              f"{np.mean(d[N // 4]):11.4e} {np.mean(d[N // 2]):11.4e} "
              f"{np.mean(d[N]):11.4e}   {e:11.4e} {gb / e if e > 0 else float('nan'):7.2f} "
              f"{np.sqrt(e) if e > 0 else float('nan'):10.3e}")

    (ART / f"ceiling{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"\n# wrote {ART / f'ceiling{args.tag}.json'}")
    return 0


# ---------------------------------------------------------------------------
def mode_anatomy(args) -> int:
    """The two premises the mixture hypothesis rests on, measured.

    (a) "the closure error at L=32 is concentrated in the top eigendirection of
        Cov(h^32)" — project the error onto the eigenbasis and read the
        cumulative share.
    (b) "the non-Gaussianity is concentrated in very few directions" — the
        participation ratio of the variance versus the participation ratio of
        the *excess kurtosis*, which is the quantity a mixture actually
        removes.
    """
    ART.mkdir(parents=True, exist_ok=True)
    out = {"mlps": [], "n_samples": args.n_pilot}
    for i in range(args.mlps):
        seed = MLP_SEED_BASE + i
        W = make_mlp(256, 32, seed)
        _, Z = collect(W, args.n_pilot, seed=5_000_000 + i)
        m0, U, ev = frame_eig(Z)
        Zd = Z.astype(np.float64)
        truth = np.maximum(Zd, 0.0).mean(axis=0)
        s = np.sqrt(np.maximum(np.diag(np.cov(Zd.T, bias=True)), 1e-30))
        err_oracle = relu_mean(m0, s) - truth
        err_chain = chain_run(W)[-1] - truth

        # (a) error energy in the eigenframe of Cov(relu(z))
        H = np.maximum(Zd, 0.0)
        Hc = H - H.mean(axis=0)
        Sh = (Hc.T @ Hc) / Zd.shape[0]
        wh, Vh = np.linalg.eigh(Sh)
        o = np.argsort(wh)[::-1]
        Vh = Vh[:, o]
        wh = wh[o]
        cum = {}
        for e, nm in ((err_oracle, "oracle"), (err_chain, "chain")):
            c = (Vh.T @ e) ** 2
            c = np.cumsum(c) / c.sum()
            cum[nm] = [float(c[j - 1]) for j in (1, 2, 4, 8, 16, 32, 64, 256)]

        # (c) the readout form itself, with the strongest oracle possible:
        #     give EVERY neuron its own optimal K-cell 1-D mixture, aligned to
        #     its own pre-activation.  No shared mixture can do this -- one
        #     partition has to serve all 256 neurons at once -- so if this is
        #     small the obstruction is the sharing, and if it is large the
        #     readout form  sum_k w_k relu_mean(m_k, s_k)  is itself the wall.
        perneuron = {}
        Zs = np.sort(Zd, axis=0)
        c1 = np.concatenate([np.zeros((1, 256)), np.cumsum(Zs, axis=0)])
        c2 = np.concatenate([np.zeros((1, 256)), np.cumsum(Zs * Zs, axis=0)])
        N = Zd.shape[0]
        for Kc in (2, 4, 6, 16, 64):
            e = np.linspace(0, N, Kc + 1).astype(np.int64)
            cnt = (e[1:] - e[:-1]).astype(np.float64)[:, None]
            mc = (c1[e[1:]] - c1[e[:-1]]) / cnt
            vc = np.maximum((c2[e[1:]] - c2[e[:-1]]) / cnt - mc * mc, 1e-30)
            pred = (cnt * relu_mean(mc, np.sqrt(vc))).sum(axis=0) / N
            perneuron[Kc] = float(np.mean((pred - truth) ** 2))
        del Zs, c1, c2

        # (b) participation ratios
        pr_var = float(ev.sum() ** 2 / (ev ** 2).sum())
        T = (Zd - m0) @ U
        t = T / np.sqrt(np.maximum(ev, 1e-30))
        kurt = np.abs((t ** 4).mean(axis=0) - 3.0)
        skew = np.abs((t ** 3).mean(axis=0))
        pr_kurt = float(kurt.sum() ** 2 / (kurt ** 2).sum())
        pr_skew = float(skew.sum() ** 2 / (skew ** 2).sum())
        rec = {
            "seed": seed,
            "ev_top1_share": float(ev[0] / ev.sum()),
            "pr_variance": pr_var,
            "pr_abs_excess_kurtosis": pr_kurt,
            "pr_abs_skew": pr_skew,
            "kurt_top1_share": float(kurt[0] / kurt.sum()),
            "kurt_of_top_eigendir": float(kurt[0]),
            "kurt_max_over_eigendirs": float(kurt.max()),
            "kurt_argmax_eigendir": int(np.argmax(kurt)),
            "err_cum_share_oracle": cum["oracle"],
            "err_cum_share_chain": cum["chain"],
            "rms_oracle": float(np.sqrt(np.mean(err_oracle ** 2))),
            "rms_chain": float(np.sqrt(np.mean(err_chain ** 2))),
            "per_neuron_oracle_mse": {str(k): v for k, v in perneuron.items()},
        }
        out["mlps"].append(rec)
        print(f"  seed {seed}: PR(var)={pr_var:.1f}  top1 var share="
              f"{rec['ev_top1_share']:.3f}  PR(|kurt|)={pr_kurt:.1f}  "
              f"kurt top1 share={rec['kurt_top1_share']:.3f}", flush=True)

    def avg(k):
        return np.mean([m[k] for m in out["mlps"]], axis=0)

    print()
    print("# premise (a): is the closure error concentrated in the top "
          "eigendirections of Cov(h^32)?")
    print(f"  {'directions':>12} {'oracle closure':>16} {'analytic chain':>16}"
          f" {'isotropic':>11}")
    for j, nd in enumerate((1, 2, 4, 8, 16, 32, 64, 256)):
        print(f"  {nd:>12} {avg('err_cum_share_oracle')[j]:>16.3f} "
              f"{avg('err_cum_share_chain')[j]:>16.3f} {nd / 256:>11.3f}")
    print()
    print("# premise (b): variance is low-rank; is the non-Gaussianity?")
    print(f"  participation ratio of the variance        {avg('pr_variance'):.1f}")
    print(f"  participation ratio of |excess kurtosis|   "
          f"{avg('pr_abs_excess_kurtosis'):.1f}")
    print(f"  participation ratio of |skewness|          {avg('pr_abs_skew'):.1f}")
    print(f"  top-1 eigendirection: share of variance    {avg('ev_top1_share'):.3f}")
    print(f"  top-1 eigendirection: share of |kurtosis|  {avg('kurt_top1_share'):.3f}")
    print()
    print("# premise (c): is the READOUT FORM the wall, or is it the sharing?")
    print("#   every neuron given its own optimal K-cell 1-D mixture on its own")
    print("#   pre-activation -- an oracle no shared mixture can reach")
    for Kc in ("2", "4", "6", "16", "64"):
        v = float(np.mean([m["per_neuron_oracle_mse"][Kc] for m in out["mlps"]]))
        print(f"  K = {Kc:>2} per neuron       raw MSE {v:.4e}   "
              f"rms {np.sqrt(v):.3e}")
    (ART / "anatomy.json").write_text(json.dumps(out, indent=1))
    print(f"\n# wrote {ART / 'anatomy.json'}")
    return 0


# ---------------------------------------------------------------------------
def mixture_predict(W, nodes, r=1, split_at=0, resplit=0, kmax=8, K=None):
    """The deployable propagator: no sampling anywhere, ``K = nodes^r``.

    ``z^1 = x W^1`` is exactly Gaussian, so a split at ``split_at = 0`` splits
    a law that is exactly right.  Splitting a Gaussian is a pure quadrature
    identity and changes nothing on its own — the components only start to
    matter once each has passed through its own rectification, after which the
    mixture's mean and covariance still agree with the single-Gaussian chain's
    at that layer but its *shape* does not, and from the next layer on the two
    answers diverge.
    """
    n = W[0].shape[1]
    W64 = [w.astype(np.float64) for w in W]
    st = MixtureState.gaussian(np.zeros(n), W64[0].T @ W64[0])
    K = K or nodes ** r
    for l in range(len(W)):
        if l == split_at and nodes > 1:
            for _ in range(r):
                st = st.split(st.top_direction(), nodes)
        elif resplit and l > split_at and (l - split_at) % resplit == 0:
            st = st.split(st.top_direction(), nodes).reduce_to(K)
        if l + 1 == len(W):
            return st.relu_mean_mixture()
        st = st.relu_step(W64[l + 1], kmax=kmax,
                          exact_acos=(l == 0 and st.K == 1))
    raise AssertionError


def mixture_kernel(weights, ctx=None, nodes=6, r=1, kmax=4, split_at=0):
    """The propagator written directly against flopscope, so what is billed is
    what is measured.  Mirrors ``kernels.cov_prop_mehler`` component by
    component; the split uses power iteration rather than an eigensolver, which
    costs ``~15 * 2n^2`` against the layer's ``n^3`` and is therefore free."""
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor.kernels import _hermite_coeffs, _relu_gauss  # noqa: PLC0415

    n = weights[0].shape[0]
    xq, wq = gauss_hermite(nodes)
    mus = [fnp.zeros(n, dtype=fnp.float32)]
    covs = [flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))]
    ws = [1.0]
    rows = []
    for li, w in enumerate(weights):
        pre = [(w.T @ mu, fnp.einsum("ij,ia,jb->ab", cov, w, w))
               for mu, cov in zip(mus, covs)]
        if li == split_at and nodes > 1:
            for _ in range(r):
                npre, nws = [], []
                for (mu_pre, cov_pre), wk in zip(pre, ws):
                    u = fnp.ones(n, dtype=fnp.float32) * fnp.float32(1.0 / n ** 0.5)
                    for _ in range(40):
                        u = cov_pre @ u
                        u = u / fnp.sqrt(fnp.sum(u * u))
                    cu = cov_pre @ u
                    s2 = fnp.maximum(fnp.sum(cu * u), 1e-12)
                    s = fnp.sqrt(s2)
                    child = cov_pre - fnp.outer(cu, cu) / s2
                    child = flops.symmetrize(child, symmetry=(0, 1))
                    for q in range(nodes):
                        npre.append((mu_pre + cu * fnp.float32(xq[q]) / s, child))
                        nws.append(wk * float(wq[q]))
                pre, ws = npre, nws
        mus, covs, acc = [], [], None
        for (mu_pre, cov_pre), wk in zip(pre, ws):
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sig = fnp.sqrt(var_pre)
            mu, var_post, alpha, ph, Ph = _relu_gauss(mu_pre, var_pre, sig)
            acc = mu * wk if acc is None else acc + mu * wk
            inv = 1.0 / sig
            rho = fnp.maximum(fnp.minimum(cov_pre * fnp.outer(inv, inv), 1.0), -1.0)
            a = _hermite_coeffs(alpha, sig, ph, Ph, kmax)
            rho_k = rho
            cv = fnp.outer(a[1], a[1]) * rho
            fact = 1.0
            for k in range(2, kmax + 1):
                rho_k = rho_k * rho
                fact *= k
                cv = cv + fnp.outer(a[k], a[k]) * (rho_k * (1.0 / fact))
            fnp.fill_diagonal(cv, var_post)
            mus.append(mu)
            covs.append(flops.symmetrize(cv, symmetry=(0, 1)))
        rows.append(acc)
    return fnp.stack(rows, axis=0)


def bill_kernel(args):
    """Billed ``F`` for each mixture size, in a real ``BudgetContext``."""
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    W = make_mlp(256, 32, MLP_SEED_BASE)
    fw = [fnp.asarray(w) for w in W]
    out = {}
    # Keyed by (K, split_at): a mixture that splits at layer 8 pays for K
    # components on 24 layers, not 32, so K alone does not determine F.
    for nodes, r, split_at, _ in PROP_PLAN:
        key = (nodes ** r, split_at)
        if key in out:
            continue
        with flops.BudgetContext(flop_budget=10 ** 13, quiet=True) as c:
            mixture_kernel(fw, nodes=nodes, r=r, kmax=args.kmax_bill,
                           split_at=split_at)
        out[key] = int(c.flops_used)
    with flops.BudgetContext(flop_budget=10 ** 13, quiet=True) as c:
        mixture_kernel(fw, nodes=1, r=1, kmax=args.kmax_bill, split_at=0)
    out[(1, 0)] = int(c.flops_used)
    return out


def cached_truth(W, i, n_half, chunk):
    """Two independent reference halves per MLP, cached on disk."""
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / f"truth_{MLP_SEED_BASE + i}_{n_half}.npz"
    if p.exists():
        d = np.load(p)
        return d["a"], d["b"]
    a = truth_mean(W, n_half, seed=8_000_000 + 2 * i, chunk=chunk)
    b = truth_mean(W, n_half, seed=8_000_000 + 2 * i + 1, chunk=chunk)
    np.savez(p, a=a, b=b)
    return a, b


#: ``(nodes, r, split_at, resplit)``.  A split at layer 1 is nearly worthless
#: and the reason is measurable: the closure covariance has participation ratio
#: 127 at ``L = 1`` with the top eigendirection carrying **1.5%** of the trace,
#: falling to 6.5 and 34% by ``L = 32``.  There is no dominant direction to
#: condition on until the network has made one, so the sweep walks the split
#: point down the depth.
PROP_PLAN = [
    (6, 1, 0, 0),
    (2, 1, 8, 0), (4, 1, 8, 0), (6, 1, 8, 0), (12, 1, 8, 0),
    (6, 1, 16, 0), (6, 1, 24, 0), (6, 1, 28, 0), (6, 1, 31, 0),
    (3, 2, 16, 0),
    (2, 1, 0, 8), (2, 1, 0, 4), (2, 1, 0, 1), (3, 1, 0, 4),
]


def mode_propagate(args) -> int:
    ART.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple, list] = {}
    base_l: list[float] = []
    for i in range(args.mlps):
        seed = MLP_SEED_BASE + i
        W = make_mlp(256, 32, seed)
        a, b = cached_truth(W, i, args.n_truth, args.chunk)
        base_l.append(umse(chain_run(W)[-1], a, b))
        for cfg in PROP_PLAN:
            t0 = time.time()
            p = mixture_predict(W, cfg[0], r=cfg[1], split_at=cfg[2],
                                resplit=cfg[3])
            rows.setdefault(cfg, []).append(umse(p, a, b))
            print(f"  seed {seed} nodes={cfg[0]:2d} r={cfg[1]} "
                  f"split@{cfg[2]:2d} resplit={cfg[3]}  "
                  f"raw={rows[cfg][-1]:.4e}  chain={base_l[-1]:.4e}  "
                  f"{time.time() - t0:.1f}s", flush=True)

    bill = bill_kernel(args)
    base = float(np.mean(base_l))
    print()
    print(f"  analytic Gaussian chain (K = 1): raw {base:.4e}")
    print()
    hdr = (f"  {'nodes':>5} {'r':>2} {'split':>5} {'resp':>4} {'K':>4} "
           f"{'raw MSE':>12} {'x chain':>8} {'F':>14} {'F/B':>7} {'adjusted':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cfg in PROP_PLAN:
        v = float(np.mean(rows[cfg]))
        K = cfg[0] ** cfg[1]
        F = bill.get((K, cfg[2]), 0)
        fb = F / 272_000_000_000
        print(f"  {cfg[0]:>5} {cfg[1]:>2} {cfg[2]:>5} {cfg[3]:>4} {K:>4} "
              f"{v:12.4e} {base / v:8.2f} {F:14,d} {fb:7.4f} "
              f"{v * max(0.1, fb):11.4e}")
    (ART / "propagate.json").write_text(json.dumps(
        {"chain": base, "bill": {f"{k[0]}|{k[1]}": v for k, v in bill.items()},
         "rows": {"|".join(map(str, k)): float(np.mean(v))
                  for k, v in rows.items()}}, indent=1))
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ceiling",
                    choices=("ceiling", "anatomy", "propagate"))
    ap.add_argument("--mlps", type=int, default=4)
    ap.add_argument("--n-pilot", type=int, default=150_000)
    ap.add_argument("--n-cells", type=int, default=1_000_000)
    ap.add_argument("--n-truth", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--k", default="2,4,6,8")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--kmax-bill", type=int, default=4)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    return {"ceiling": mode_ceiling, "anatomy": mode_anatomy,
            "propagate": mode_propagate}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
