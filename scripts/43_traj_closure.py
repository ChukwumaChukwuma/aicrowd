"""Trajectory-calibrated moment closure: can `E[relu(z^L)]` be computed 4.4x
better than the Gaussian closure?

`docs/integrable_cv.md` reduced the whole control-variate programme to one
scalar: ``r``, the factor by which an analytic layer-mean beats the Gaussian
closure's.  Break-even against the shipped basis is ``r = 4.2-4.6`` at every
depth from 6 to 32.  This script measures ``r`` for

* the Gaussian closure itself (``r = 1`` by definition),
* this repository's ``kappa_3`` star-diagram arm and the full tree catalogue,
* a **trajectory-calibrated** closure -- per-layer linear corrections fitted on
  the chain's own rolled-forward trajectories, DAgger-style, after
  jamesrahenry's method (forum topic 18097, submission #314695, MIT replication
  repository ``jamesrahenry/arc-whitebox-replication``),
* and the oracle ceilings of that family (per-MLP fitted coefficients; a
  per-layer full-rank correction), which bound every feature map inside it.

The yardstick is fixed before any run: ``rms(mu_closure^L - mu_true^L)`` against
a large independent Monte-Carlo reference, at ``L = 8, 16, 24, 32``, with
``mu^L = E[relu(z^L)]``.  Reference error is removed by the unbiased two-half
cross estimator ``E[(mt - mA)(mt - mB)] = (mt - m)^2``, so no figure here is
inflated by the reference's own noise.

Modes
-----
``ref``       cache the Monte-Carlo reference (per-layer mean and mean-square,
              in two independent halves) for a list of MLPs.
``baseline``  ``r`` for every analytic arm already in the repository.
``probe``     what a per-layer linear correction can explain, layer by layer,
              teacher-forced and rolled-forward.
``fit``       DAgger fit of the per-layer coefficients on the train split.
``eval``      ``r`` of the fitted closure on the held-out split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.official_seeds import make_official_mlp  # noqa: E402

ART = Path(os.environ.get("WHEST_ARTIFACTS", "artifacts"))
REFDIR = ART / "traj"
CHUNK = 4096

#: The depths the pre-registered bar is read at.
BAR_LAYERS = (8, 16, 24, 32)
#: Pre-registered acceptance bar on ``r`` (docs/integrable_cv.md sec 3.2).
BAR_R = 4.4


def mlp_weights(seed: int, official: bool = False):
    if official:
        return make_official_mlp(WIDTH, DEPTH, seed)
    return make_mlp(WIDTH, DEPTH, seed)


def load_official_seeds():
    from whestfloor.suite import Suite  # noqa: PLC0415
    return list(Suite.load(ART / "suites" / "official_mini.npz").mlp_seeds)


# ---------------------------------------------------------------------------
# mode: ref -- the Monte-Carlo reference, in two independent halves
# ---------------------------------------------------------------------------
def reference_path(seed: int, official: bool) -> Path:
    tag = "off" if official else "loc"
    return REFDIR / f"ref_{tag}_{seed}.npz"


def make_reference(seed: int, n_samples: int, official: bool,
                   want_cov: bool = False) -> dict:
    """Per-layer ``E[h^l]`` and ``E[(h^l)^2]``, accumulated in two halves.

    The two halves use disjoint sample streams, so for any deterministic
    ``mt`` the product ``(mt - mA)(mt - mB)`` is an unbiased estimator of the
    squared error against the population mean.
    """
    W = mlp_weights(seed, official)
    n = WIDTH
    sh = np.zeros((2, DEPTH, n))
    sh2 = np.zeros((2, DEPTH, n))
    cnt = [0, 0]
    HH = np.zeros((DEPTH, n, n)) if want_cov else None
    rng = np.random.default_rng((abs(seed) % 1_000_000_007) * 7 + 13)
    half_n = n_samples // 2
    for half in (0, 1):
        done = 0
        while done < half_n:
            m = min(CHUNK, half_n - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            h = x
            for l in range(DEPTH):
                h = np.maximum(h @ W[l], 0.0)
                sh[half, l] += h.sum(0, dtype=np.float64)
                sh2[half, l] += np.einsum("ij,ij->j", h, h, dtype=np.float64)
                if want_cov and half == 0:
                    HH[l] += (h.T.astype(np.float64) @ h.astype(np.float64))
            done += m
        cnt[half] = half_n
    out = {
        "seed": np.int64(seed), "n": np.int64(half_n),
        "mA": sh[0] / half_n, "mB": sh[1] / half_n,
        "qA": sh2[0] / half_n, "qB": sh2[1] / half_n,
    }
    if want_cov:
        out["HH"] = HH / half_n
    return out


def mode_ref(seeds, n_samples, official, want_cov):
    REFDIR.mkdir(parents=True, exist_ok=True)
    for k, seed in enumerate(seeds):
        p = reference_path(seed, official)
        if p.exists():
            print(f"[{k+1}/{len(seeds)}] {seed}: cached")
            continue
        t0 = time.time()
        d = make_reference(seed, n_samples, official, want_cov)
        np.savez_compressed(p, **d)
        mt = 0.5 * (d["mA"] + d["mB"])
        noise = float(np.sqrt(np.mean((d["mA"] - d["mB"]) ** 2) / 4.0))
        print(f"[{k+1}/{len(seeds)}] {seed}: {time.time()-t0:.0f}s  "
              f"rms m^32 {np.sqrt(np.mean(mt[-1]**2)):.4f}  "
              f"reference noise (per half-pair) {noise:.2e}", flush=True)


def load_reference(seed: int, official: bool) -> dict:
    p = reference_path(seed, official)
    if not p.exists():
        raise SystemExit(f"missing reference {p}; run --mode ref first")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# mode: aux -- the auxiliary targets and oracle inputs
#   * ``HH[l] = E[h^l (h^l)']``          -- the off-diagonal fit target
#   * ``k3[l], k4[l]``                   -- exact marginal cumulants of z^{l+1}
#   * ``kiij[l] = E[zc_i^2 zc_j]``       -- the exact pair field
# The last two are jamesrahenry's probe inputs at infinite probe size, i.e. his
# "perfect-cumulant control": an upper bound on every probe-fed variant.
# ---------------------------------------------------------------------------
def aux_path(seed: int, official: bool) -> Path:
    tag = "off" if official else "loc"
    return REFDIR / f"aux_{tag}_{seed}.npz"


def make_aux(seed: int, n_samples: int, official: bool) -> dict:
    W = mlp_weights(seed, official)
    n = WIDTH
    sz = np.zeros((DEPTH, n))
    sz2 = np.zeros((DEPTH, n))
    sz3 = np.zeros((DEPTH, n))
    sz4 = np.zeros((DEPTH, n))
    ZZZ = np.zeros((DEPTH, n, n))
    HH = np.zeros((DEPTH, n, n))
    sh = np.zeros((DEPTH, n))
    rng = np.random.default_rng((abs(seed) % 1_000_000_007) * 31 + 5)
    done = 0
    while done < n_samples:
        mchunk = min(CHUNK, n_samples - done)
        x = rng.standard_normal((mchunk, n), dtype=np.float32)
        h = x
        for l in range(DEPTH):
            z = h @ W[l]
            z64 = z.astype(np.float64)
            sz[l] += z64.sum(0)
            z2 = z64 * z64
            sz2[l] += z2.sum(0)
            sz3[l] += (z2 * z64).sum(0)
            sz4[l] += (z2 * z2).sum(0)
            ZZZ[l] += z2.T @ z64           # E[z_i^2 z_j], centred later
            h = np.maximum(z, 0.0)
            h64 = h.astype(np.float64)
            sh[l] += h64.sum(0)
            HH[l] += h64.T @ h64
        done += mchunk
    N = float(done)
    mz = sz / N
    m2 = sz2 / N
    m3 = sz3 / N
    m4 = sz4 / N
    v = m2 - mz * mz
    k3 = m3 - 3.0 * mz * m2 + 2.0 * mz ** 3
    k4 = m4 - 4.0 * mz * m3 - 3.0 * m2 * m2 + 12.0 * mz * mz * m2 - 6.0 * mz ** 4
    # E[zc_i^2 zc_j] = E[z_i^2 z_j] - a_j E[z_i^2] - 2 a_i E[z_i z_j]
    #                  + 2 a_i^2 a_j.
    # ``E[z^{l+1} (z^{l+1})'] = W' E[h^l (h^l)'] W`` is an EXACT identity, so
    # the second cross-moment needs no accumulation of its own.
    mh = sh / N
    kiij = np.zeros((DEPTH, n, n))
    for l in range(DEPTH):
        a = mz[l]
        Wl = W[l].astype(np.float64)
        if l == 0:
            Ezz = Wl.T @ Wl
        else:
            Ezz = Wl.T @ (HH[l - 1] / N) @ Wl
        kiij[l] = (ZZZ[l] / N - a[None, :] * m2[l][:, None]
                   - 2.0 * a[:, None] * Ezz + 2.0 * (a * a)[:, None] * a[None, :])
    return {
        "seed": np.int64(seed), "n": np.int64(done),
        "mz": mz, "vz": v, "k3": k3, "k4": k4,
        "kiij": kiij.astype(np.float32),
        "HH": (HH / N).astype(np.float32),
        "mh": mh,
    }


def mode_aux(seeds, n_samples, official):
    REFDIR.mkdir(parents=True, exist_ok=True)
    for k, seed in enumerate(seeds):
        p = aux_path(seed, official)
        if p.exists():
            print(f"[{k+1}/{len(seeds)}] {seed}: cached")
            continue
        t0 = time.time()
        d = make_aux(seed, n_samples, official)
        np.savez(p, **d)
        g3 = d["k3"] / d["vz"] ** 1.5
        print(f"[{k+1}/{len(seeds)}] {seed}: {time.time()-t0:.0f}s  "
              f"gamma3 rms L8/L16/L32 "
              f"{np.sqrt(np.mean(g3[7]**2)):.3f}/"
              f"{np.sqrt(np.mean(g3[15]**2)):.3f}/"
              f"{np.sqrt(np.mean(g3[31]**2)):.3f}", flush=True)


def load_aux(seed: int, official: bool) -> dict:
    p = aux_path(seed, official)
    if not p.exists():
        raise SystemExit(f"missing aux {p}; run --mode aux first")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# The yardstick.  ``r`` is a ratio of rms errors of ``E[relu(z^L)]``, and the
# reference's own noise is removed exactly: with two independent halves,
# ``E[(mt - mA)(mt - mB)] = (mt - m_pop)^2`` for any deterministic ``mt``.
# ---------------------------------------------------------------------------
def unbiased_mse(mt, ref, layer):
    """Unbiased ``mean_j (mt_j - m_pop_j)^2`` at 1-indexed layer ``layer``."""
    eA = mt - ref["mA"][layer - 1]
    eB = mt - ref["mB"][layer - 1]
    return float(np.mean(eA * eB))


def naive_mse(mt, ref, layer):
    mm = 0.5 * (ref["mA"][layer - 1] + ref["mB"][layer - 1])
    return float(np.mean((mt - mm) ** 2))


ARMS = {
    # name: kwargs to whestfloor.trajclosure.chain, plus fixed coefficients
    "gauss": dict(src="coinc"),
    "k3star": dict(src="star", mspec=("k3E",), fixed=(1.0,)),
    "k3tree": dict(src="tree", mspec=("k3E",), fixed=(1.0,)),
    "k34tree": dict(src="tree", mspec=("k3E", "k4E"), fixed=(1.0, 1.0)),
    "k34coinc": dict(src="coinc", mspec=("k3E", "k4E"), fixed=(1.0, 1.0)),
}


def run_arm(W, spec, depth=DEPTH):
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    kw = dict(spec)
    fixed = kw.pop("fixed", None)
    mspec = kw.get("mspec", ())
    coefs = None
    if fixed is not None:
        Am = np.asarray(fixed, dtype=np.float64)
        coefs = [(Am, np.zeros(0), np.zeros(0)) for _ in range(depth)]
    return tc.run(W, coefs=coefs, **kw), mspec


def mode_baseline(seeds, official, arms, layers):
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    print("# r for every analytic arm already in the repository.\n"
          "# r_L = rms(mu_gauss^L - mu_true^L) / rms(mu_arm^L - mu_true^L),\n"
          "# unbiased two-half reference estimator, mu^L = E[relu(z^L)].\n"
          f"# {len(seeds)} MLPs, {'official' if official else 'local'}.\n")
    per = {a: {L: [] for L in layers} for a in arms}
    for k, seed in enumerate(seeds):
        ref = load_reference(seed, official)
        W = [w.astype(np.float64) for w in mlp_weights(seed, official)]
        for a in arms:
            mus, _ = run_arm(W, ARMS[a])
            for L in layers:
                per[a][L].append(unbiased_mse(mus[L - 1], ref, L))
        print(f"  [{k+1}/{len(seeds)}] {seed} done", flush=True)
    print(f"\n  {'arm':>10} " + "".join(f"{'L=%d' % L:>12}" for L in layers)
          + "   " + "".join(f"{'r@%d' % L:>8}" for L in layers))
    base = {L: float(np.mean(per["gauss"][L])) for L in layers}
    rows = []
    for a in arms:
        cells, rs = [], []
        for L in layers:
            v = float(np.mean(per[a][L]))
            cells.append(f"{math.sqrt(max(v, 0)):12.4e}")
            rs.append(f"{math.sqrt(base[L] / max(v, 1e-30)):8.3f}")
        print(f"  {a:>10} " + "".join(cells) + "   " + "".join(rs))
        rows.append({"arm": a,
                     "rms": {L: math.sqrt(max(float(np.mean(per[a][L])), 0))
                             for L in layers},
                     "r": {L: math.sqrt(base[L] / max(float(np.mean(per[a][L])),
                                                      1e-30)) for L in layers},
                     "per_mlp_mse": {L: per[a][L] for L in layers}})
    tag = "off" if official else "loc"
    (ART / "traj").mkdir(parents=True, exist_ok=True)
    (ART / "traj" / f"baseline_{tag}.json").write_text(
        json.dumps({"seeds": [int(s) for s in seeds], "rows": rows}, indent=1))
    return rows


# ---------------------------------------------------------------------------
# mode: ceiling -- oracle re-anchoring, which bounds every correction family
# ---------------------------------------------------------------------------
def anchored_chain(W, ref, aux, *, mean=False, postvar=False, prevar=False,
                   postcov=False):
    """Roll the Gaussian closure forward, overwriting parts of the state with
    Monte-Carlo truth at every layer boundary.

    ``mean``     ``E[relu(z^l)] <- truth``.  A perfect per-layer mean
                 correction is exactly this, so its final error is the CEILING
                 of every mean-correction family (jamesrahenry sec 1b's
                 "reset mean only").
    ``postvar``  ``Var(relu(z^l)) <- truth``  ("variances only").
    ``postcov``  the whole ``Cov(relu(z^l)) <- truth`` ("full covariance").
    ``prevar``   rescale the pre-activation covariance so its diagonal is the
                 true ``Var(z^l)``, keeping the chain's correlation.
    """
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    n = WIDTH
    m = np.zeros(n)
    C = W[0].T @ W[0]
    out = []
    for l in range(DEPTH):
        if prevar:
            sc = np.sqrt(aux["vz"][l] / np.maximum(np.diag(C), 1e-30))
            C = C * np.outer(sc, sc)
        mu0, Ch0, sig, al, pa, Pa, rho, ac = tc.closure_step(
            m, C, exact_acos=(l == 0 and not prevar))
        out.append(mu0)
        mu = ref["mean_true"][l] if mean else mu0
        Ch = Ch0
        if postcov:
            Ch = aux["HH"][l].astype(np.float64) - np.outer(
                ref["mean_true"][l], ref["mean_true"][l])
        elif postvar:
            Ch = Ch0.copy()
            np.fill_diagonal(Ch, ref["var_true"][l])
        if l + 1 < DEPTH:
            Wn = W[l + 1]
            m = Wn.T @ mu
            C = Wn.T @ Ch @ Wn
    return np.stack(out)


CEILINGS = {
    "gauss": {},
    "mean": dict(mean=True),
    "postvar": dict(postvar=True),
    "postcov": dict(postcov=True),
    "mean+postvar": dict(mean=True, postvar=True),
    "mean+postcov": dict(mean=True, postcov=True),
    "mean+prevar": dict(mean=True, prevar=True),
}


def mode_ceiling(seeds, official, layers, names):
    need_aux = any(("postcov" in CEILINGS[k] or "prevar" in CEILINGS[k])
                   for k in names)
    print("# Oracle re-anchoring: what a PERFECT per-layer correction reaches.\n"
          "# Every row overwrites part of the chain's state with Monte-Carlo\n"
          "# truth at every layer boundary, then reads the closure's own mean.\n"
          f"# {len(seeds)} MLPs, {'official' if official else 'local'}.\n")
    acc = {k: {L: [] for L in layers} for k in names}
    for k, seed in enumerate(seeds):
        net = load_net(seed, official, need_aux)
        _, W, ref, aux = net
        for nm in names:
            mus = anchored_chain(W, ref, aux, **CEILINGS[nm])
            for L in layers:
                acc[nm][L].append(unbiased_mse(mus[L - 1], ref, L))
        print(f"  [{k+1}/{len(seeds)}] {seed} done", flush=True)
    base = {L: float(np.mean(acc["gauss"][L])) for L in layers}
    print(f"\n  {'oracle':>14} " + "".join(f"{'L=%d' % L:>12}" for L in layers)
          + "   " + "".join(f"{'r@%d' % L:>8}" for L in layers))
    rows = []
    for nm in names:
        v = {L: float(np.mean(acc[nm][L])) for L in layers}
        print(f"  {nm:>14} "
              + "".join(f"{math.sqrt(max(v[L], 0)):12.4e}" for L in layers)
              + "   "
              + "".join(f"{math.sqrt(base[L] / max(v[L], 1e-30)):8.3f}"
                        for L in layers))
        rows.append({"oracle": nm, "mse": v,
                     "r": {L: math.sqrt(base[L] / max(v[L], 1e-30))
                           for L in layers},
                     "per_mlp": {L: acc[nm][L] for L in layers}})
    tag = "off" if official else "loc"
    (ART / "traj").mkdir(parents=True, exist_ok=True)
    (ART / "traj" / f"ceiling_{tag}.json").write_text(
        json.dumps({"seeds": [int(s) for s in seeds], "rows": rows}, indent=1))
    return rows


# ---------------------------------------------------------------------------
# The trajectory-calibrated fit (jamesrahenry's method, topic 18097)
# ---------------------------------------------------------------------------
#: Named feature packages.  ``free`` is the deployable one: nothing in it needs
#: a Monte-Carlo probe of the target network.
PACKS = {
    # probe-free, no cumulant features at all
    "shape": dict(mspec=("sphi", "sphi_a", "sphi_a2"),
                  vspec=("vphi", "vphi_a", "vphi_a2"), ospec=()),
    # probe-free, analytic cumulants from the diagram catalogue
    "free": dict(mspec=("k3E", "k4E", "sphi", "sphi_a", "sphi_a2",
                        "k3E_a", "k4E_a", "k3n"),
                 vspec=("k3V", "vphi", "vphi_a", "vphi_a2", "k3n_v", "k4n_v"),
                 ospec=("T1", "T1rho")),
    # probe-free, mean+var only (no covariance correction)
    "freem": dict(mspec=("k3E", "k4E", "sphi", "sphi_a", "sphi_a2",
                         "k3E_a", "k4E_a", "k3n"),
                  vspec=("k3V", "vphi", "vphi_a", "vphi_a2", "k3n_v", "k4n_v"),
                  ospec=()),
    # a wider probe-free mean head
    "wide": dict(mspec=("k3E", "k4E", "sphi", "sphi_a", "sphi_a2", "sphi_a3",
                        "k3E_a", "k4E_a", "k3n", "sPhi", "mu0"),
                 vspec=("k3V", "vphi", "vphi_a", "vphi_a2", "k3n_v", "k4n_v"),
                 ospec=("T1", "T1rho")),
    # MEAN ONLY.  The covariance is left exactly as the Gaussian closure
    # produces it -- see sec 1(b) of jamesrahenry's write-up and the
    # ``varoracle`` row of ``--mode ceiling``: re-anchoring the covariance to
    # truth makes the chain WORSE, because the covariance error is what
    # compensates the mean error downstream.
    "m3": dict(mspec=("sphi", "sphi_a", "sphi_a2"), vspec=(), ospec=()),
    "m8": dict(mspec=("k3E", "k4E", "sphi", "sphi_a", "sphi_a2",
                      "k3E_a", "k4E_a", "k3n"), vspec=(), ospec=()),
    "m12": dict(mspec=("k3E", "k4E", "k3E_a", "k4E_a", "k3n",
                       "sphi", "sphi_a", "sphi_a2", "sphi_a3",
                       "sPhi", "mu0", "one"), vspec=(), ospec=()),
    # mean + off-diagonal covariance, no diagonal
    "m8o": dict(mspec=("k3E", "k4E", "sphi", "sphi_a", "sphi_a2",
                       "k3E_a", "k4E_a", "k3n"), vspec=(),
                ospec=("T1", "T1rho")),
    "m12o": dict(mspec=("k3E", "k4E", "k3E_a", "k4E_a", "k3n",
                        "sphi", "sphi_a", "sphi_a2", "sphi_a3",
                        "sPhi", "mu0", "one"), vspec=(),
                 ospec=("T1", "T1rho")),
}


def solve(X, y, lam=0.0):
    """Pooled least squares with an optional relative ridge.

    ``lstsq``'s pseudo-inverse, not ``solve``: at layer 1 the pre-activation is
    exactly Gaussian so every cumulant feature is identically zero and the
    normal matrix is singular by construction.  That is the correct answer
    (those features get coefficient 0), not an error.
    """
    if X.shape[1] == 0:
        return np.zeros(0)
    G = X.T @ X
    b = X.T @ y
    if lam:
        G = G + lam * (np.trace(G) / max(X.shape[1], 1)) * np.eye(X.shape[1])
    return np.linalg.lstsq(G, b, rcond=None)[0]


def oracle_cum_list(aux):
    """``ext_cum`` for :func:`whestfloor.trajclosure.chain` from exact MC."""
    return [(aux["k3"][l], aux["k4"][l], aux["kiij"][l].astype(np.float64))
            for l in range(DEPTH)]


def probe_cum_list(W, n_probe, seed):
    """jamesrahenry's ``N``-sample plain-MC probe, verbatim in structure."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_probe, WIDTH), dtype=np.float32)
    out = []
    for l in range(DEPTH):
        z = (x @ W[l]).astype(np.float64)
        mu = z.mean(0)
        zc = z - mu
        s2 = (zc * zc).mean(0)
        out.append(((zc ** 3).mean(0),
                    (zc ** 4).mean(0) - 3.0 * s2 * s2,
                    (zc * zc).T @ zc / n_probe))
        x = np.maximum(z, 0.0).astype(np.float32)
    return out


def fit_dagger(train, pack, *, src, lam, transport=None, cum="analytic",
               n_probe=4096, verbose=True):
    """Sequential per-layer least squares on rolled-forward states.

    ``train`` is a list of ``(seed, W, ref, aux)``.  At layer ``l`` every train
    net is advanced with the corrections of layers ``0..l-1`` already applied,
    the design matrices are built at the state actually visited, and one pooled
    OLS gives layer ``l``'s coefficients.  That is the whole of the method: the
    training distribution is the deployment distribution.
    """
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    n = WIDTH
    mspec, vspec, ospec = pack["mspec"], pack["vspec"], pack["ospec"]
    want_off = bool(ospec)
    gens = []
    for (seed, W, ref, aux) in train:
        ext = None
        if cum == "oracle":
            ext = oracle_cum_list(aux)
        elif cum == "probe":
            ext = probe_cum_list(W, n_probe, abs(int(seed)) % 100000 + 777)
        gens.append(tc.ChainDriver(W, mspec=mspec, vspec=vspec, ospec=ospec,
                                   src=src, ext_cum=ext, want_off=want_off,
                                   transport=transport))
    coefs = []
    iu = np.triu_indices(n, k=1)
    hist = []
    for l in range(DEPTH):
        recs = [g.step() for g in gens]
        Xm, ym, Xv, yv, Xo, yo = [], [], [], [], [], []
        for (seed, W, ref, aux), rec in zip(train, recs):
            tmu = ref["mean_true"][l]
            Xm.append(rec["Fm"])
            ym.append(tmu - rec["mu0"])
            if vspec:
                tv = ref["var_true"][l]
                Xv.append(rec["Fv"])
                yv.append(tv - np.diag(rec["Ch0"]))
            if ospec and rec["Fo"]:
                tC = aux["HH"][l].astype(np.float64) - np.outer(tmu, tmu)
                Xo.append(np.stack([f[iu] for f in rec["Fo"]], axis=1))
                yo.append((tC - rec["Ch0"])[iu])
        Am = solve(np.concatenate(Xm), np.concatenate(ym), lam)
        Av = solve(np.concatenate(Xv), np.concatenate(yv), lam) if Xv \
            else np.zeros(0)
        Ao = solve(np.concatenate(Xo), np.concatenate(yo), lam) if Xo \
            else np.zeros(0)
        coefs.append((Am, Av, Ao))
        for g in gens:
            g.apply(Am, Av, Ao)
        if verbose and (l + 1) in (1, 4, 8, 16, 24, 32):
            e0 = np.mean([np.mean((r["mu0"] - t[2]["mean_true"][l]) ** 2)
                          for r, t in zip(recs, train)])
            e1 = np.mean([np.mean((g.last_mu - t[2]["mean_true"][l]) ** 2)
                          for g, t in zip(gens, train)])
            r2 = 1.0 - e1 / max(e0, 1e-300)
            hist.append({"L": l + 1, "mse_raw": float(e0),
                         "mse_corr": float(e1), "R2": float(r2)})
            print(f"    L={l+1:>2}  train closure {math.sqrt(e0):.3e} -> "
                  f"corrected {math.sqrt(max(e1,0)):.3e}   "
                  f"(step R^2 {r2*100:6.2f}%,  x {math.sqrt(e0/max(e1,1e-300)):5.2f})",
                  flush=True)
    return coefs, hist


def eval_coefs(nets, pack, coefs, *, src, layers, transport=None,
               cum="analytic", n_probe=4096):
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    out = {L: [] for L in layers}
    for (seed, W, ref, aux) in nets:
        ext = None
        if cum == "oracle":
            ext = oracle_cum_list(aux)
        elif cum == "probe":
            ext = probe_cum_list(W, n_probe, abs(int(seed)) % 100000 + 4321)
        mus = tc.run(W, coefs=coefs, src=src, ext_cum=ext,
                     want_off=bool(pack["ospec"]), transport=transport,
                     **{k: pack[k] for k in ("mspec", "vspec", "ospec")})
        for L in layers:
            out[L].append(unbiased_mse(mus[L - 1], ref, L))
    return out


def load_net(seed, official, need_aux):
    ref = load_reference(seed, official)
    ref["mean_true"] = 0.5 * (ref["mA"] + ref["mB"])
    ref["var_true"] = np.maximum(
        0.5 * (ref["qA"] + ref["qB"]) - ref["mean_true"] ** 2, 1e-30)
    aux = load_aux(seed, official) if need_aux else None
    W = [w.astype(np.float64) for w in mlp_weights(seed, official)]
    return (seed, W, ref, aux)


def mode_fit(train_seeds, eval_seeds, official, pack_name, src, lam,
             layers, cum, n_probe, transport, tag):
    pack = PACKS[pack_name]
    need_aux = bool(pack["ospec"]) or cum == "oracle"
    print(f"# trajectory-calibrated closure: pack={pack_name} src={src} "
          f"cum={cum} lam={lam} transport={transport}\n"
          f"# train {len(train_seeds)} MLPs, eval {len(eval_seeds)} MLPs "
          f"({'official' if official else 'local'})\n")
    t0 = time.time()
    train = [load_net(s, official, need_aux) for s in train_seeds]
    ev = [load_net(s, official, need_aux) for s in eval_seeds]
    print(f"  loaded [{time.time()-t0:.0f}s]", flush=True)
    coefs, hist = fit_dagger(train, pack, src=src, lam=lam,
                             transport=transport, cum=cum, n_probe=n_probe)
    res = eval_coefs(ev, pack, coefs, src=src, layers=layers,
                     transport=transport, cum=cum, n_probe=n_probe)
    base = eval_coefs(ev, dict(mspec=(), vspec=(), ospec=()), None,
                      src="coinc", layers=layers)
    print(f"\n  {'L':>4} {'gauss rms':>12} {'traj rms':>12} {'r':>8}   "
          f"{'r (median)':>11}")
    rows = []
    for L in layers:
        b = float(np.mean(base[L]))
        v = float(np.mean(res[L]))
        rmed = math.sqrt(np.median(np.asarray(base[L]) /
                                   np.maximum(res[L], 1e-30)))
        rows.append({"L": L, "mse_gauss": b, "mse_traj": v,
                     "r": math.sqrt(b / max(v, 1e-30)), "r_median": rmed,
                     "per_mlp_gauss": base[L], "per_mlp_traj": res[L]})
        print(f"  {L:>4} {math.sqrt(max(b,0)):12.4e} {math.sqrt(max(v,0)):12.4e} "
              f"{math.sqrt(b/max(v,1e-30)):8.3f}   {rmed:11.3f}")
    rmin = min(r["r"] for r in rows)
    print(f"\n  pre-registered bar: r > {BAR_R} at every L in {BAR_LAYERS}.  "
          f"min r = {rmin:.3f}  ->  "
          f"{'PASS' if rmin > BAR_R else 'FAIL'}")
    (ART / "traj").mkdir(parents=True, exist_ok=True)
    (ART / "traj" / f"fit_{tag}.json").write_text(json.dumps({
        "pack": pack_name, "src": src, "cum": cum, "lam": lam,
        "n_probe": n_probe, "transport": transport,
        "train": [int(s) for s in train_seeds],
        "eval": [int(s) for s in eval_seeds],
        "train_hist": hist, "rows": rows,
        "coefs": [[list(map(float, c)) for c in lay] for lay in coefs],
    }, indent=1))
    return rows


# ---------------------------------------------------------------------------
# mode: stepfit -- the ceiling of the WHOLE programme
#
# Anchor the state to truth at every layer (so the chain contributes nothing),
# then fit the richest per-neuron mean correction the family allows, pooled
# across the train MLPs, and read it out on held-out ones.  Nothing a
# trajectory-calibrated closure can do beats "perfect state tracking + the best
# per-layer per-neuron correction", so this row bounds the programme.
# ---------------------------------------------------------------------------
def rich_mean_basis(sig, alpha, pa, Pa, extra=()):
    cols = [sig * pa * alpha ** k for k in range(6)]
    cols += [sig * Pa, sig, np.ones_like(sig)]
    cols += list(extra)
    return np.stack(cols, axis=1)


def mode_stepfit(train_seeds, eval_seeds, official, layers, use_oracle_cum):
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    print("# THE CEILING OF THE PROGRAMME.  The chain's state is overwritten\n"
          "# with truth at every layer, so only the one-step closure error is\n"
          "# left; then the richest per-neuron mean correction the family\n"
          "# allows is fitted, pooled over the train MLPs, and read out on\n"
          "# held-out ones.  No trajectory calibration can beat this.\n"
          f"# oracle cumulants in the basis: {use_oracle_cum}\n")
    nets = {s: load_net(s, official, True) for s in train_seeds + eval_seeds}

    def states(seed):
        """(mu0, residual, basis) at every layer, at the exact input state."""
        _, W, ref, aux = nets[seed]
        n = WIDTH
        m = np.zeros(n)
        C = W[0].T @ W[0]
        out = []
        for l in range(DEPTH):
            mu0, Ch0, sig, al, pa, Pa, rho, ac = tc.closure_step(
                m, C, exact_acos=(l == 0))
            extra = ()
            if use_oracle_cum:
                k3, k4 = aux["k3"][l], aux["k4"][l]
                var = sig * sig
                extra = (-(k3 / 6.0) * (al * pa / var),
                         (k4 / 24.0) * (pa * (al * al - 1.0) / (sig * var)),
                         -(k3 / 6.0) * (al * al * pa / var),
                         (k4 / 24.0) * (pa * al * (al * al - 1.0) / (sig * var)))
            B = rich_mean_basis(sig, al, pa, Pa, extra)
            out.append((mu0, ref["mean_true"][l] - mu0, B))
            if l + 1 < DEPTH:
                Wn = W[l + 1]
                mt = ref["mean_true"][l]
                Cht = aux["HH"][l].astype(np.float64) - np.outer(mt, mt)
                m = Wn.T @ mt
                C = Wn.T @ Cht @ Wn
        return out

    tr = {s: states(s) for s in train_seeds}
    ev = {s: states(s) for s in eval_seeds}
    res = {L: [] for L in layers}
    raw = {L: [] for L in layers}
    nai = {L: [] for L in layers}
    noise = []
    for l in range(DEPTH):
        X = np.concatenate([tr[s][l][2] for s in train_seeds])
        y = np.concatenate([tr[s][l][1] for s in train_seeds])
        beta = solve(X, y)
        if l + 1 in res:
            L = l + 1
            for s in eval_seeds:
                ref = nets[s][2]
                mu0, r0, B = ev[s][l]
                res[L].append(unbiased_mse(mu0 + B @ beta, ref, L))
                raw[L].append(unbiased_mse(mu0, ref, L))
                nai[L].append(naive_mse(mu0 + B @ beta, ref, L))
                noise.append(float(np.mean((ref["mA"][L - 1]
                                            - ref["mB"][L - 1]) ** 2)) / 8.0)
    nz = float(np.mean(noise))
    print(f"  {'L':>4} {'one-step rms':>13} {'+ fitted corr':>14} "
          f"{'step gain':>10}   {'(naive rms)':>12}")
    for L in layers:
        a0 = float(np.mean(raw[L]))
        a1 = float(np.mean(res[L]))
        an = float(np.mean(nai[L])) - nz
        if a1 > nz:
            cell = f"{math.sqrt(a0 / a1):10.3f}        "
            val = math.sqrt(a1)
        elif an > nz:
            cell = f"{math.sqrt(a0 / an):10.3f} (floor)"
            val = math.sqrt(an)
        else:
            cell = f"{'>' + '%.1f' % math.sqrt(a0 / nz):>10} (floor)"
            val = math.sqrt(nz)
        print(f"  {L:>4} {math.sqrt(max(a0,0)):13.4e} {val:14.4e} {cell}   "
              f"{math.sqrt(max(float(np.mean(nai[L])),0)):12.4e}")
    print(f"  reference noise on the averaged mean: {math.sqrt(nz):.2e} rms")
    return raw, res


# ---------------------------------------------------------------------------
# mode: fidelity -- how accurately would the cumulant field have to be known?
#
# The one-step ceiling of sec 5 is unlocked by the per-neuron kappa_3/kappa_4
# field and by nothing else.  Degrade the TRUE field to a controlled R^2 and
# read the one-step gain back.  This converts "we cannot predict the cumulants"
# into a number a future scheme can be held to.
# ---------------------------------------------------------------------------
def mode_fidelity(train_seeds, eval_seeds, official, layers, r2_grid):
    from whestfloor import trajclosure as tc  # noqa: PLC0415
    print("# One-step closure error at the EXACT second-order state, corrected\n"
          "# by the Edgeworth terms of a kappa_3/kappa_4 field known to a\n"
          "# controlled fraction R2 of its own variance (the rest is replaced\n"
          "# by an independent draw with the matched spectrum).\n"
          f"# per-layer coefficients fitted on {len(train_seeds)} MLPs, read "
          f"out on {len(eval_seeds)} held out.\n")
    rng = np.random.default_rng(20260807)

    def collect(seed):
        _, W, ref, aux = load_net(seed, official, True)
        m = np.zeros(WIDTH)
        C = W[0].T @ W[0]
        rows = {}
        for l in range(DEPTH):
            mu0, Ch0, sig, al, pa, Pa, rho, ac = tc.closure_step(
                m, C, exact_acos=(l == 0))
            L = l + 1
            if L in layers:
                var = sig * sig
                B0 = rich_mean_basis(sig, al, pa, Pa)
                per_q = {}
                for q in r2_grid:
                    kk = []
                    for k in (aux["k3"][l], aux["k4"][l]):
                        z = rng.standard_normal(WIDTH)
                        z = z / max(np.std(z), 1e-30) * np.std(k)
                        kk.append(math.sqrt(q) * k
                                  + math.sqrt(max(1 - q, 0.0)) * z)
                    k3, k4 = kk
                    ex = (-(k3 / 6.0) * (al * pa / var),
                          (k4 / 24.0) * (pa * (al * al - 1.0) / (sig * var)),
                          -(k3 / 6.0) * (al * al * pa / var),
                          (k4 / 24.0) * (pa * al * (al * al - 1.0) / (sig * var)))
                    per_q[q] = np.concatenate([B0, np.stack(ex, axis=1)],
                                              axis=1)
                rows[L] = (mu0, ref["mean_true"][l] - mu0, per_q, ref)
            if l + 1 < DEPTH:
                Wn = W[l + 1]
                mt = ref["mean_true"][l]
                Cht = aux["HH"][l].astype(np.float64) - np.outer(mt, mt)
                m = Wn.T @ mt
                C = Wn.T @ Cht @ Wn
        return rows

    tr = {s: collect(s) for s in train_seeds}
    ev = {s: collect(s) for s in eval_seeds}
    base = {L: float(np.mean([unbiased_mse(ev[s][L][0], ev[s][L][3], L)
                              for s in eval_seeds])) for L in layers}
    #: the reference's own noise, below which a residual cannot be resolved
    floor = float(np.mean([np.mean((ev[s][layers[0]][3]["mA"]
                                    - ev[s][layers[0]][3]["mB"]) ** 2) / 4.0
                           for s in eval_seeds]))
    print(f"  {'R2 of the cumulant field':>26} "
          + "".join(f"{'gain L=%d' % L:>12}" for L in layers))
    print(f"  {'0 (no cumulants at all)':>26} "
          + "".join(f"{1.0:12.2f}" for L in layers))
    for q in r2_grid:
        cells = []
        for L in layers:
            X = np.concatenate([tr[s][L][2][q] for s in train_seeds])
            y = np.concatenate([tr[s][L][1] for s in train_seeds])
            beta = solve(X, y)
            v = float(np.mean([
                unbiased_mse(ev[s][L][0] + ev[s][L][2][q] @ beta,
                             ev[s][L][3], L) for s in eval_seeds]))
            if v < floor:
                cells.append(f"{'>' + '%.1f' % math.sqrt(base[L]/floor):>12}")
            else:
                cells.append(f"{math.sqrt(base[L] / v):12.2f}")
        print(f"  {q:26.4f} " + "".join(cells))
    print(f"\n# '>' marks a residual below the reference's own noise floor "
          f"({math.sqrt(floor):.2e} rms); the measurement stops there.")
    print("\n# for scale, the best available predictors of that field:\n"
          "#   analytic star / tree diagrams  R^2 = 0.00-0.12 at depth\n"
          "#   3-factor model on Cov(z^L)     R^2 = 0.78-0.81\n"
          "#   degree-5 polynomial in alpha   R^2 = 0.84-0.87")


# ---------------------------------------------------------------------------
# mode: price -- what a given r is actually worth on the graded score
# ---------------------------------------------------------------------------
#: ``docs/integrable_cv.md`` sec 3.1, official ladder (2 MLPs, --mode ladder
#: --official): ``L -> (R2_eff, rms projected bias of the Gaussian closure)``.
#: The bias is ``c'(mtilde - m)`` at the ridge coefficients that would actually
#: be used, NOT ``rms(mtilde - m)``; dividing it by ``r`` is the same operation
#: that page's ``--mode summary`` performs.
LADDER_OFFICIAL = {
    1: (0.4136, 5.9e-05), 2: (0.5022, 1.42e-03), 4: (0.5962, 3.16e-03),
    8: (0.6966, 4.22e-03), 16: (0.8360, 5.33e-03), 24: (0.9295, 5.91e-03),
    32: (0.9887, 6.36e-03),
}
LADDER_LOCAL = {
    1: (0.3773, 8.8e-05), 2: (0.4615, 1.56e-03), 4: (0.5821, 3.36e-03),
    8: (0.7164, 5.26e-03), 16: (0.8596, 6.02e-03), 24: (0.9383, 6.68e-03),
    32: (0.9886, 7.00e-03),
}
V0 = 4.06e-07          # 0.1 V/N at the shipped operating point
SHIP_ADJ = 2.47e-07    # the shipped adjusted score the ladder is scored against
BUDGET = 2.72e11       # B
CLAMP_BUDGET = 0.1 * BUDGET


def price(r_by_L, ladder, closure_flops=0.0):
    """`x ship` of the layer-``L`` control variate at closure accuracy ``r``.

    ``MSE(theta) = (1-R1) V/N - 2 theta D + theta^2 (D + b^2)`` is minimised at
    ``theta* = D/(D+b^2)`` and saves ``D^2/(D+b^2)``.  The closure's own FLOPs
    are charged where they are actually paid: at the multiplier clamp the
    sample budget is ``0.1B - F_closure``, so ``V/N`` inflates by
    ``1/(1 - F_closure/0.1B)``.
    """
    VN = V0 / 0.1
    R1 = ladder[1][0]
    infl = 1.0 / max(1.0 - closure_flops / CLAMP_BUDGET, 1e-9)
    out = []
    for L, r in sorted(r_by_L.items()):
        R2, b0 = ladder[L]
        D = max(R2 - R1, 0.0) * VN * infl
        b2 = (b0 / r) ** 2
        mse = (1 - R1) * VN * infl - (D * D / (D + b2) if D > 0 else 0.0)
        out.append({"L": L, "r": r, "theta": D / (D + b2) if D + b2 else 0.0,
                    "adjusted": 0.1 * mse, "x_ship": SHIP_ADJ / (0.1 * mse)})
    return out


def mode_price(r_grid, layers, ladder_name, closure_flops):
    ladder = LADDER_OFFICIAL if ladder_name == "official" else LADDER_LOCAL
    print("# What a closure accuracy r is worth, on docs/integrable_cv.md's\n"
          f"# exact objective, {ladder_name} ladder, closure charged "
          f"{closure_flops:.2e} FLOPs = "
          f"{100*closure_flops/CLAMP_BUDGET:.1f}% of the clamp budget.\n")
    print("  " + f"{'r':>7}" + "".join(f"{'L=%d' % L:>10}" for L in layers)
          + f"{'best':>10}")
    for r in r_grid:
        rows = {p["L"]: p for p in price({L: r for L in layers}, ladder,
                                         closure_flops)}
        best = max(rows.values(), key=lambda p: p["x_ship"])
        print(f"  {r:7.2f}" + "".join(f"{rows[L]['x_ship']:10.3f}"
                                      for L in layers)
              + f"{best['x_ship']:9.3f}x @ L={best['L']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("ref", "aux", "baseline", "ceiling", "fit",
                             "stepfit", "fidelity", "price"))
    ap.add_argument("--r2-grid", default="0.5,0.8,0.9,0.95,0.99,1.0")
    ap.add_argument("--oracle-cum-basis", action="store_true")
    ap.add_argument("--r-grid", default="1,1.6,2,2.9,4.4,5.85,7.2,10")
    ap.add_argument("--ladder", default="official",
                    choices=("official", "local"))
    ap.add_argument("--closure-flops", type=float, default=0.0)
    ap.add_argument("--ceilings", default="gauss,mean,postvar,mean+postvar")
    ap.add_argument("--pack", default="free")
    ap.add_argument("--src", default="star")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--cum", default="analytic",
                    choices=("analytic", "oracle", "probe"))
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--transport", default="")
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--aux-samples", type=int, default=400_000)
    ap.add_argument("--arms", default="gauss,k3star,k3tree,k34tree")
    ap.add_argument("--layers", default="8,16,24,32")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--seed0", type=int, default=960_000)
    ap.add_argument("--mlps", type=int, default=16)
    ap.add_argument("--ref-samples", type=int, default=2_000_000)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--want-cov", action="store_true")
    a = ap.parse_args()

    if a.seeds:
        seeds = [int(v) for v in a.seeds.split(",")]
    elif a.official:
        seeds = load_official_seeds()[: a.mlps]
    else:
        seeds = [a.seed0 + i for i in range(a.mlps)]

    layers = [int(v) for v in a.layers.split(",")]
    transport = None
    if a.transport:
        transport = tuple(float(v) for v in a.transport.split(","))
    if a.mode == "ref":
        mode_ref(seeds, a.ref_samples, a.official, a.want_cov)
    elif a.mode == "aux":
        mode_aux(seeds, a.aux_samples, a.official)
    elif a.mode == "baseline":
        mode_baseline(seeds, a.official, a.arms.split(","), layers)
    elif a.mode == "ceiling":
        mode_ceiling(seeds, a.official, layers, a.ceilings.split(","))
    elif a.mode == "stepfit":
        mode_stepfit(seeds[: a.train], seeds[a.train:], a.official, layers,
                     a.oracle_cum_basis)
    elif a.mode == "fidelity":
        mode_fidelity(seeds[: a.train], seeds[a.train:], a.official, layers,
                      [float(v) for v in a.r2_grid.split(",")])
    elif a.mode == "price":
        mode_price([float(v) for v in a.r_grid.split(",")],
                   [L for L in (2, 4, 8, 16, 24, 32)], a.ladder,
                   a.closure_flops)
    elif a.mode == "fit":
        mode_fit(seeds[: a.train], seeds[a.train:], a.official, a.pack, a.src,
                 a.lam, layers, a.cum, a.n_probe, transport, a.tag)


if __name__ == "__main__":
    main()
