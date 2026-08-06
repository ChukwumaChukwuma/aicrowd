#!/usr/bin/env python
"""Why network-space surrogates decorrelate: the weight-perturbation law.

``scripts/22`` measures rank-truncated surrogates and finds them useless.  This
script isolates the *mechanism* so the kill generalises past the particular
truncation used, and turns it into a number every future surrogate proposal can
be checked against in one line.

Construction.  Perturb every weight matrix by a fixed relative amount while
preserving its norm exactly in expectation,

    W~_l = sqrt(1 - d^2) W_l + d G_l ,   G_l ~ iid N(0, 2/n) ,

so ``d`` is the relative perturbation with no gain-collapse confound (the
failure mode that makes an unrescaled SVD truncation output literal zeros).
Drive ``f`` and ``f~`` with the SAME input and measure

    rho(d) = corr( f(x)_j , f~(x)_j )      averaged over neurons,
    V(d)   = Var( f(x)_j - f~(x)_j )       averaged over neurons.

MLMC needs ``V/V_full`` at the top level below ~5e-2 just to pay for itself, and
below ~1e-3 to clear a 20x bar, i.e. ``rho >= 0.9995``.

The second half of the script measures what ``d`` a rank-r surrogate actually
achieves, from two spectra:

* the singular values of ``W`` (what plain SVD truncation can keep), and
* the eigenvalues of the measured activation second moment ``E[h h^T]`` at every
  layer (what an input-adapted projector can keep) — the more favourable of the
  two, and the one that decides the family.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.official_seeds import make_official_mlp  # noqa: E402

WIDTH = 256
DEPTH = 32


def forward_pair(x, W, Wt):
    """Propagate the same input through both networks; return both outputs."""
    h, g = x, x
    for a, b in zip(W, Wt):
        h = np.maximum(h @ a, 0.0, dtype=np.float32)
        g = np.maximum(g @ b, 0.0, dtype=np.float32)
    return h, g


def perturb(W, d, rng):
    n = W[0].shape[0]
    s = float(np.sqrt(2.0 / n))
    keep = float(np.sqrt(max(0.0, 1.0 - d * d)))
    return [(keep * w + d * (rng.standard_normal(w.shape) * s)).astype(np.float32)
            for w in W]


def act_spectra(W, n_samples, chunk, seed):
    """Cumulative capture of ``E||h||^2`` by the top-r eigenspace of E[h h^T]."""
    n = W[0].shape[0]
    M = [np.zeros((n, n)) for _ in range(len(W))]
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        h = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(W):
            hd = h.astype(np.float64)
            M[li] += hd.T @ hd
            h = np.maximum(h @ w, 0.0, dtype=np.float32)
        done += nb
    caps = []
    for m in M:
        ev = np.linalg.eigvalsh(m / done)[::-1]
        ev = np.maximum(ev, 0.0)
        caps.append(np.cumsum(ev) / ev.sum())
    return np.array(caps)          # (depth, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mlps", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--deltas", type=str,
                    default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1")
    ap.add_argument("--suite", type=str,
                    default=os.path.join(os.environ.get("WHEST_ARTIFACTS", "."),
                                         "suites", "official_mini.npz"))
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",") if x]
    z = np.load(args.suite, allow_pickle=False)
    seeds = [int(s) for s in z["mlp_seeds"][: args.n_mlps]]

    print(f"# weight-perturbation decorrelation, {len(seeds)} official MLPs, "
          f"N={args.n_samples} coupled samples")
    print()

    acc = {d: [] for d in deltas}
    vfull = []
    wspec_cap = []
    aspec_cap = []
    for i, ms in enumerate(seeds):
        W = make_official_mlp(WIDTH, DEPTH, ms)
        # singular-value capture of W itself
        sv = np.array([np.linalg.svd(w.astype(np.float64), compute_uv=False)
                       for w in W])
        wspec_cap.append((np.cumsum(sv ** 2, axis=1)
                          / (sv ** 2).sum(axis=1, keepdims=True)).mean(axis=0))
        aspec_cap.append(act_spectra(W, 4096, args.chunk, 4242 + i))

        prng = np.random.default_rng(31337 + i)
        Wt = {d: perturb(W, d, prng) for d in deltas}
        n = WIDTH
        S = {d: np.zeros(n) for d in deltas}
        Q = {d: np.zeros(n) for d in deltas}
        SD = {d: np.zeros(n) for d in deltas}
        QD = {d: np.zeros(n) for d in deltas}
        Sf = np.zeros(n)
        Qf = np.zeros(n)
        SP = {d: np.zeros(n) for d in deltas}
        rng = np.random.default_rng(777_000 + i)
        done = 0
        while done < args.n_samples:
            nb = min(args.chunk, args.n_samples - done)
            x = rng.standard_normal((nb, n), dtype=np.float32)
            h = x
            for w in W:
                h = np.maximum(h @ w, 0.0, dtype=np.float32)
            hd = h.astype(np.float64)
            Sf += hd.sum(axis=0)
            Qf += (hd * hd).sum(axis=0)
            for d in deltas:
                g = x
                for w in Wt[d]:
                    g = np.maximum(g @ w, 0.0, dtype=np.float32)
                gd = g.astype(np.float64)
                S[d] += gd.sum(axis=0)
                Q[d] += (gd * gd).sum(axis=0)
                SP[d] += (hd * gd).sum(axis=0)
                df = hd - gd
                SD[d] += df.sum(axis=0)
                QD[d] += (df * df).sum(axis=0)
            done += nb
        N = float(done)
        mf = Sf / N
        vf = (Qf / N - mf ** 2)
        vfull.append(vf.mean())
        for d in deltas:
            mg = S[d] / N
            vg = Q[d] / N - mg ** 2
            cv = SP[d] / N - mf * mg
            md = SD[d] / N
            vd = QD[d] / N - md ** 2
            ok = (vf > 0) & (vg > 0)
            rho = np.zeros(n)
            rho[ok] = cv[ok] / np.sqrt(vf[ok] * vg[ok])
            acc[d].append((float(vd.mean() / vf.mean()), float(rho.mean()),
                           float((md ** 2).mean() / (mf ** 2).mean())))
        print(f"  mlp {i} done", flush=True)

    V = float(np.mean(vfull))
    print()
    print(f"full-network per-neuron variance V = {V:.6f}")
    print()
    print("   delta      Var(f-f~)/V     rho        rel bias^2")
    print("   " + "-" * 52)
    rows = []
    for d in deltas:
        a = np.mean(acc[d], axis=0)
        rows.append((d, a[0], a[1], a[2]))
        print(f"   {d:9.1e}  {a[0]:13.4e}  {a[1]:9.6f}  {a[2]:11.3e}")

    # local slope: V/V_full ~ K * delta^2 in the perturbative regime
    d0, r0 = rows[0][0], rows[0][1]
    K = r0 / (d0 ** 2)
    print()
    print(f"   perturbative law  Var(f-f~)/V ~ K d^2  with K = {K:.3e}")
    for tgt, label in ((5e-2, "MLMC break-even"), (1e-3, "20x bar")):
        print(f"   -> d required for V/V_full = {tgt:.0e} ({label}): "
              f"{np.sqrt(tgt / K):.3e}")

    wc = np.mean(wspec_cap, axis=0)
    ac = np.mean(aspec_cap, axis=0)          # (depth, n)
    print()
    print("== what rank buys what perturbation ==")
    print("   d_rank(r) = sqrt(1 - captured fraction)")
    print()
    print("     r    W spectrum   d_W      act.spectrum(L16)  d_act   "
          "act.spectrum(L32)  d_act")
    for r in (1, 2, 4, 8, 16, 32, 64, 128, 192, 240, 255):
        cw = wc[r - 1]
        c16 = ac[15, r - 1]
        c32 = ac[31, r - 1]
        print(f"   {r:5d}  {cw:10.6f}  {np.sqrt(max(0,1-cw)):7.4f}  "
              f"{c16:17.9f}  {np.sqrt(max(0,1-c16)):6.4f}  "
              f"{c32:17.9f}  {np.sqrt(max(0,1-c32)):6.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"V": V, "rows": rows, "K": float(K),
                   "w_capture": wc.tolist(),
                   "act_capture": ac.tolist(),
                   "mlp_seeds": seeds, "n_samples": args.n_samples},
                  open(args.out, "w"))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
