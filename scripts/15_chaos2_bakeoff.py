#!/usr/bin/env python
"""Do the chaos-2 or conditional-independence *distributions* beat a Gaussian?

The cumulant route is dead (ledger: ``reconstruction_from_cumulants``) because a
truncated cumulant vector does not determine ``E[relu]``.  Both models tested
here avoid moments entirely: each is a genuine, valid distribution for ``S_j``.

* **chaos-2** — expand every rectifier in Hermite polynomials of its own
  standardised pre-activation and keep chaos orders 0-2 exactly.  Chaos 1 +
  chaos 2 is a *generalized chi-square*; chaos 3+ is replaced by an independent
  Gaussian of the correct variance.  See :mod:`whestfloor.chaos`.
* **cond-indep** — the exact CGF of a sum of *independent* rectified Gaussians,
  affinely renormalised so its first two moments match the exact ones taken from
  the full covariance, so the correlations only have to be right in their effect
  on the shape.  See :mod:`whestfloor.condindep`.

**Bar, fixed before the run:** a model must beat the Gaussian model by **30x in
RMS** on the one-step error.  The whole oracle-cumulant family manages only
~10x (ledger: ``edgeworth_oracle_cumulants``), so anything below 30x is not a
new mechanism, just a repackaging of the old one.

What is measured
----------------

A *real* layer: an MLP is propagated with measured moments, so ``R`` is the
correlation the network actually produces.  ``z`` is then drawn **exactly**
Gaussian from that measured ``(m, Sigma)``, which isolates the error every model
here attacks — "``S_j`` is not Gaussian even when ``z`` is" — from the separate
error "``z`` is not Gaussian".  Both are reported.

Truth is measured with a chaos-2 control variate::

    truth = chaos2_exact + E[Y - X],
    Y = relu(S(u)),  X = E_eta[relu(P(u) + eta)],  E[X] = chaos2_exact

where ``P`` is the exact chaos-0/1/2 part of ``S`` evaluated pathwise.  So
``E[Y-X]`` **is** the chaos-2 error, measured directly and at ~25x lower
variance than a plain Monte Carlo, and any other model's error is
``E[Y-X] + (chaos2 - model)`` with the bracket analytic.  Two independent halves
give an unbiased MSE for every model, with the reference's own noise removed
exactly (same estimator as :mod:`whestfloor.harness`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.chaos import (  # noqa: E402
    brute_mc, chaos2_relu_mean, chaos_pieces, cumulants_from_central,
    gauss_relu_mean, layer_state,
)
from whestfloor.condindep import (  # noqa: E402
    indep_relu_mean_fft, indep_std_cumulants,
)
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import hermite_prob, phi, relu_mean  # noqa: E402

BAR_FACTOR = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--layers", type=str, default="1,3,5")
    ap.add_argument("--n-state", type=int, default=400_000)
    ap.add_argument("--n-brute", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--kmax", type=int, default=120)
    ap.add_argument("--nj", type=int, default=0,
                    help="score only the first NJ outputs (the FFT model is the "
                         "expensive one); 0 = all")
    ap.add_argument("--log2m", type=int, default=18)
    ap.add_argument("--xis", type=str, default="",
                    help="oracle damping factors for the quadratic form")
    ap.add_argument("--no-fft", action="store_true")
    a = ap.parse_args()

    W = make_mlp(a.width, a.depth, a.seed)
    print(f"# chaos-2 / cond-indep bake-off  width={a.width} depth={a.depth} "
          f"mlp_seed={a.seed}")
    print(f"# n_state={a.n_state:,}  n_brute={a.n_brute:,} per half x 2 halves")
    print(f"# BAR fixed before the run: beat the Gaussian model by {BAR_FACTOR:.0f}x in RMS")
    print()

    worst = {}
    for li in [int(x) for x in a.layers.split(",")]:
        t0 = time.time()
        m, cov, netmean = layer_state(W, li, a.n_state, 4242 + li, want_next=True)
        s = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        R = cov / np.outer(s, s)
        np.fill_diagonal(R, 1.0)
        Wn = (W[li + 1] if li + 1 < a.depth else W[0]).astype(np.float64)
        n = a.width
        jj = np.arange(n) if a.nj <= 0 else np.arange(a.nj)

        pc = chaos_pieces(Wn, m, s, R, kmax=a.kmax)
        pg = gauss_relu_mean(pc)
        rep: dict = {}
        p2 = chaos2_relu_mean(Wn, m, s, R, pc, report=rep)

        ya, da, ba, _, ma = brute_mc(Wn, m, cov, pc, a.n_brute, 900_000 + 2 * li)
        yb, db, bb, _, mb = brute_mc(Wn, m, cov, pc, a.n_brute, 900_001 + 2 * li)
        da, db = da + ba, db + bb
        truth = p2 + 0.5 * (da + db)
        _, k2t, k3t, k4t = cumulants_from_central(0.5 * (ma + mb))

        models = {"gauss": pg,
                  "gauss(oracle v)": relu_mean(pc["c"], np.sqrt(k2t)),
                  "chaos2": p2}
        for xi in (float(x) for x in a.xis.split(",") if x):
            pcx = dict(pc)
            pcx["G2"] = pc["G2"] * xi
            pcx["v_eta"] = np.maximum(
                pc["v_tot"] - pc["v_l"] - xi * xi * pc["v_q"], 1e-30)
            models[f"chaos2(xi={xi})"] = chaos2_relu_mean(Wn, m, s, R, pcx)
        # oracle Edgeworth with the EXACT cumulants -- the ledger's reference route
        sig = np.sqrt(k2t)
        al = pc["c"] / sig
        ph = phi(al)
        e3 = -(k3t / 6.0) * (1.0 / k2t) * hermite_prob(1, al) * ph
        e4 = (k4t / 24.0) * (sig / k2t) * hermite_prob(2, al) * ph
        models["edge3(oracle)"] = pg + e3
        models["edge4(oracle)"] = pg + e3 + e4
        if not a.no_fft:
            pi = indep_relu_mean_fft(Wn, m, s, pc["c"], pc["v_tot"],
                                     log2m=a.log2m, jobs=list(jj))
            full = np.array(pg, copy=True)
            full[jj] = pi[jj]
            models["cond-indep"] = full

        g1i, g2i, _ = indep_std_cumulants(Wn, m, s)
        off = np.abs(R - np.eye(n)).sum() / (n * (n - 1))
        print(f"layer {li + 1}  |rho|_off={off:.4f}  var frac "
              f"L={pc['v_l'].mean() / pc['v_tot'].mean():.3f} "
              f"Q={pc['v_q'].mean() / pc['v_tot'].mean():.3f} "
              f"eta={pc['v_eta'].mean() / pc['v_tot'].mean():.3f}   n_j={len(jj)}")
        print(f"   true    gamma1 rms {np.sqrt(np.mean((k3t / k2t ** 1.5) ** 2)):.4f}"
              f"  gamma2 mean {np.mean(k4t / k2t ** 2):+.4f}")
        print(f"   indep   gamma1 rms {np.sqrt(np.mean(g1i[jj] ** 2)):.4f}"
              f"  gamma2 mean {np.mean(g2i[jj]):+.4f}")
        dv = pc["v_tot"] / k2t - 1.0
        print(f"   controls: v_tot/k2_true-1 mean {dv.mean():+.1e}  "
              f"cf_tail {rep['cf_tail_bound']:.0e}  erf-bias "
              f"{np.sqrt(np.mean(ba ** 2)):.0e}  |truth-rawMC| "
              f"{np.sqrt(np.mean((truth - 0.5 * (ya + yb)) ** 2)):.1e}  "
              f"half-noise {np.std(0.5 * (da - db)):.1e}")

        base = None
        for name, p in models.items():
            ea, eb = da + (p2 - p), db + (p2 - p)
            T = float(np.mean(ea[jj] * eb[jj]))
            se = float(np.std(ea[jj] * eb[jj]) / np.sqrt(len(jj)))
            if base is None:
                base = max(T, 1e-300)
            fac = np.sqrt(base / max(T, 1e-300))
            worst.setdefault(name, []).append(fac)
            print(f"   {name:16s} unbiased MSE {T:+.4e} +-{se:.1e}   "
                  f"RMS {np.sqrt(max(T, 0.0)):.4e}   x{fac:6.2f}")

        e_b = truth - pg
        e_a = netmean - truth
        print(f"   one-step split: errB (S non-Gaussian | z Gaussian) "
              f"{np.sqrt(np.mean(e_b ** 2)):.3e}   errA (z non-Gaussian) "
              f"{np.sqrt(np.mean(e_a ** 2)):.3e}   total "
              f"{np.sqrt(np.mean((netmean - pg) ** 2)):.3e}   "
              f"[state-MC se ~{0.3 / np.sqrt(a.n_state):.0e}]")
        print(f"   [{time.time() - t0:.0f}s]")
        print()

    print("summary (best factor over the Gaussian model, per model):")
    ok = False
    for name, fs in worst.items():
        b = max(fs)
        ok |= (name in ("chaos2", "cond-indep")) and b >= BAR_FACTOR
        print(f"   {name:16s} best {b:6.2f}x   per layer "
              + " ".join(f"{f:.2f}" for f in fs))
    print()
    print("VERDICT:", "PASS" if ok else "FAIL",
          f"- bar was {BAR_FACTOR:.0f}x over the Gaussian model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
