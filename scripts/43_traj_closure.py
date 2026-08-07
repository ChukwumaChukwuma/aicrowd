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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("ref", "aux", "baseline", "ceiling", "fit"))
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
    elif a.mode == "fit":
        mode_fit(seeds[: a.train], seeds[a.train:], a.official, a.pack, a.src,
                 a.lam, layers, a.cum, a.n_probe, transport, a.tag)


if __name__ == "__main__":
    main()
