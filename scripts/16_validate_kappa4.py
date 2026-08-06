#!/usr/bin/env python
"""Validate the complete tree catalogue for kappa_3 AND kappa_4 against
brute-force Monte Carlo, in the style of scripts/13.

Bars fixed before the run:
  * kappa_3 tree catalogue: >= 95% captured (the shipped star form gets 84-88%);
  * kappa_4 tree catalogue: >= 70% captured.

Captured = ``1 - ||predicted - mc|| / ||mc||`` on a REAL layer, so ``R`` carries
the correlation structure the network actually produces.  The Monte Carlo draws
``z`` Gaussian with the measured ``(m, cov)`` -- which is exactly what the
catalogue assumes -- so what this measures is diagram truncation, NOT the
cumulant-transport deficit of ``docs/cumulant_expansion.md`` sec 9.6.  That is a
separate, larger error and is measured end to end by ``scripts/12``.

The catalogue is separately pinned to the generic diagram engine of
``scripts/03_theory_orders.py`` at 1e-15 relative by ``--pin``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.cumulants import (  # noqa: E402
    kappa3_star,
    kappa3_tree,
    kappa34_brute,
    kappa4_tree,
)
from whestfloor.mc import make_mlp  # noqa: E402


def layer_state(W, upto, n_samples, seed, chunk=1024):
    """Measured (m, cov) of z at layer `upto` (0-based), by Monte Carlo."""
    n = W[0].shape[0]
    rng = np.random.default_rng(seed)
    s1 = np.zeros(n)
    g = np.zeros((n, n))
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        for li, w in enumerate(W):
            z = x @ w
            if li == upto:
                zf = z.astype(np.float64)
                s1 += zf.sum(axis=0)
                g += zf.T @ zf
                break
            x = np.maximum(z, np.float32(0.0))
        done += nb
    m = s1 / done
    return m, g / done - np.outer(m, m)


def pin(n=48, seed=11):
    """Pin both closed forms against the generic diagram engine."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "thord", os.path.join(here, "03_theory_orders.py"))
    th = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(th)
    m, s, R, W = th.make_case(n, seed)
    ok = True
    print("# closed forms vs the generic diagram engine (trees, exact injectivity)")
    for K2, T in [(3, 3), (6, 3), (10, 4)]:
        per, _ = th.layer_cumulants(m, s, R, W, {2: K2, 3: T, 4: min(T, 3)},
                                    exact_inj=True, max_cycles=0, rs=(3, 4))
        k3e = sum(v for k, v in per.items() if k.startswith("r3"))
        k3f = kappa3_tree(W, m, s, R, K2=K2, T3=T)
        r3 = np.abs(k3e - k3f).max() / np.abs(k3e).max()
        print(f"  kappa_3  K2={K2:2d} T3={T}: rel {r3:.2e}  "
              f"[{'ok' if r3 < 1e-12 else 'FAIL'}]")
        ok &= r3 < 1e-12
        if T <= 3:
            per, _ = th.layer_cumulants(m, s, R, W, {2: K2, 3: T, 4: T},
                                        exact_inj=True, max_cycles=0, rs=(4,))
            k4e = sum(v for k, v in per.items() if k.startswith("r4"))
            k4f = kappa4_tree(W, m, s, R, K2=K2, T4=T)
            r4 = np.abs(k4e - k4f).max() / np.abs(k4e).max()
            print(f"  kappa_4  K2={K2:2d} T4={T}: rel {r4:.2e}  "
                  f"[{'ok' if r4 < 1e-12 else 'FAIL'}]")
            ok &= r4 < 1e-12
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--layers", type=str, default="1,3,5,7")
    ap.add_argument("--n-state", type=int, default=200_000)
    ap.add_argument("--n-brute", type=int, default=40_000_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args()

    if args.pin and not pin():
        print("\nVERDICT: FAIL - closed forms do not reproduce the engine")
        return 1
    print()

    W = make_mlp(args.width, args.depth, args.seed)
    print(f"# tree catalogue vs brute force   width={args.width} "
          f"depth={args.depth} n_brute={args.n_brute:,}")
    print("# bars: kappa_3 >= 0.95 captured, kappa_4 >= 0.70 captured")
    print()
    hdr = ("layer  |rho|off   ||k3||     star_u1   tree3 |   ||k4||     "
           "tree4(T4=2)  tree4(T4=3)")
    print(hdr)
    print("-" * len(hdr))

    ok3 = ok4 = True
    for li in [int(x) for x in args.layers.split(",")]:
        m, cov = layer_state(W, li, args.n_state, 4242 + li)
        s = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        R = cov / np.outer(s, s)
        np.fill_diagonal(R, 1.0)
        Wn = (W[li + 1] if li + 1 < args.depth else W[0]).astype(np.float64)

        k3t, k4t = kappa34_brute(Wn, m, cov, args.n_brute, 909 + li)
        n3, n4 = np.linalg.norm(k3t), np.linalg.norm(k4t)
        off = np.abs(R - np.eye(len(R))).sum() / (len(R) * (len(R) - 1))

        c_star = 1 - np.linalg.norm(kappa3_star(Wn, m, s, R, umax=1) - k3t) / n3
        c_tree = 1 - np.linalg.norm(kappa3_tree(Wn, m, s, R, K2=8, T3=4) - k3t) / n3
        c4 = [1 - np.linalg.norm(kappa4_tree(Wn, m, s, R, K2=8, T4=t) - k4t) / n4
              for t in (2, 3)]
        ok3 &= c_tree >= 0.95
        ok4 &= max(c4) >= 0.70
        print(f"{li+1:5d}  {off:8.4f}  {n3:9.3e}  {c_star:7.3f}  {c_tree:7.3f} | "
              f"{n4:9.3e}  {c4[0]:9.3f}  {c4[1]:11.3f}")

    print()
    print("VERDICT kappa_3:", "PASS" if ok3 else "FAIL", "(bar 0.95)")
    print("VERDICT kappa_4:", "PASS" if ok4 else "FAIL", "(bar 0.70)")
    return 0 if (ok3 and ok4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
