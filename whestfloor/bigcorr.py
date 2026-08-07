"""The offline-trained corrector, scaled: rich features, big heads, learning curves.

Why this file exists separately from :mod:`whestfloor.corrector`
----------------------------------------------------------------
``corrector.py`` is frozen around the SHIPPED 15-float head: its feature order
is indexed by ``submission/corrector.npz`` and every ablation table in
``docs/learned_corrector.md`` addresses it by position.  This module is the
scaled successor and is free to move.  Three things change.

1. **The operating point.**  The shipped head was fitted on data generated at
   ``(tau, N, P) = (2.5, 8500, 150)`` and is deployed at ``(2.5, 25000, 225)``.
   That is not a small mismatch.  The control-variate columns and the target
   both scale as ``N^-1/2``, so their ratio is invariant -- but ``one``, ``s``,
   ``Phi``, ``phi``, ``a`` do NOT scale, so their fitted coefficients are
   ``sqrt(25000/8500) = 1.72x`` too large at deployment, and they enter as a
   pure additive bias.  Everything here is generated at the shipped point.

2. **Seed replication.**  The reference ``E[relu(z^32)]`` depends only on the
   weights, so it can be amortised over ``K`` independent estimator seeds.  One
   dense reference pass costs about as much as six scored passes, so ``K = 3``
   buys three times the rows for 1.5x the compute.  The rows of one MLP share a
   reference-noise draw, which matters for the seed-INDEPENDENT columns and not
   at all for the control-variate columns; ``--mode curve`` measures both.

3. **The feature set is a superset and every channel is ablatable.**  The
   design question this module exists to answer is which per-neuron features
   carry signal, measured before any parameter count is grown.

The channels, and what is new
-----------------------------
``base``
    What ships: ``mu``, the layer-1 Hermite control variates ``cv1``/``cv2``,
    the mean-field propagated layer-1 mean gap ``cv1mf``, the final-layer shape
    ``(s, Phi, phi, alpha)`` and ``dpilot``.

``herm``
    ``cv3``, ``cv4``.  At ``N = 8500`` each Hermite block cost ``p/N = 3.0%``
    of the residual in estimation noise against ~2% of real gain, so ``k <= 2``
    was the operating point.  At ``N = 25000`` the same blocks cost **1.0%**.
    The arithmetic that closed ``k >= 3`` was N-dependent and N has tripled.

``mfv`` -- NEW, and free
    A two-channel first-order perturbation propagation.  ``z^1`` is exactly
    Gaussian, so BOTH of

        dmu1_i = mean_s relu(z^1_i) - sigma_i/sqrt(2 pi)
        dv1_i  = mean_s relu(z^1_i)^2 - sigma_i^2/2

    are exactly mean zero and free (the forward pass already has ``relu(z^1)``).
    ``cv1mf`` propagates the first and drops the second.  Both propagate
    through the same linearised closure, using

        d E[relu(z)] = Phi dm + phi ds
        d Var(relu(z)) = 2 mu0 (1 - Phi) dm + 2 (s Phi - mu0 phi) ds

    (both exact derivatives of the Gaussian rectifier moments; ``dE[relu^2]/dm
    = 2 E[relu]`` and ``dE[relu^2]/ds = 2 s Phi``), with the pre-activation
    recursions ``dm_{l+1} = W' dmu_l`` (exact) and ``dvz_{l+1} = (W .^ 2)'
    dvh_l`` (diagonal-only, an approximation that costs efficiency and never
    bias, because the propagated object is a linear functional of two exactly
    mean-zero inputs whatever the coefficients are).

    Cost: 31 extra matvecs and one elementwise square of each weight matrix --
    ``1.2e7`` FLOPs, 0.017% of the scored pass at ``N = 25000``.

``gate``
    The same propagation with the rectifier gates taken from a SUBSAMPLE of the
    scored pass instead of from the pilot.  ``Phi(alpha)`` is the propagation
    coefficient and the pilot estimates ``alpha`` from 225 samples; 4,096 rows
    of the scored pass estimate it 4x better for 1.6e8 FLOPs.  The coefficient
    is then weakly correlated with the gap it multiplies, which is an
    ``O(1/N)`` bias against an ``O(N^-1/2)`` correction -- 0.6% at this ``N``,
    i.e. 4e-5 of the variance.

``shape``
    ``sd_mc = sqrt(Var(relu z^32)/N)``, the per-neuron Monte-Carlo noise scale,
    plus ``gam1``/``gam2``.  ``sd_mc`` was dropped from the shipped 15 because a
    LINEAR head cannot use it: the optimal shrinkage of a control variate is a
    RATIO of signal to noise and a linear head has one coefficient per column.
    A nonlinear head can, which is the first concrete thing capacity buys here.

``relu1``, ``q2``
    ``docs/integrable_cv.md``'s two exactly-integrable blocks, imported from
    :mod:`whestfloor.corrector` unchanged.

``wt``
    Per-neuron final-layer weight-column moments, and per-MLP scalars.

Reference
---------
:func:`reference` is a dense float64-accumulated Monte-Carlo pass with the
input-linear (``k = 1``) control variate applied in split-sample form.  The
control variate is exactly mean zero, so the reference stays unbiased, and it
buys a measured variance reduction at no extra forward passes -- an effective
sample multiplier, which is the binding constraint on the whole programme.
"""

from __future__ import annotations

import math

import numpy as np

from . import corrector as C
from .relu_moments import relu_var

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
VAR_FLOOR = 1e-12

#: Rows of the scored draw used to re-estimate the propagation gates.
GATE_ROWS = 4096

#: Kink-frame rank for the ``q2`` block (``docs/integrable_cv.md``).
Q2_K = 32


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
#: Per-neuron arrays produced for every (MLP, estimator seed) pair.
PER_SEED: tuple[str, ...] = (
    "mu",
    "cv1", "cv2", "cv3",
    "cv1mf", "cv1mfg",
    "mfm", "mfv", "mfv2", "mfmg", "mfvg", "mfv2g",
    "relu1", "q2",
    "s", "Phi", "phi", "alpha", "sd_mc", "gam1", "gam2",
    "dpilot", "alpha_p", "u1", "u2",
)
#: Per-(MLP, seed) scalars.
PER_SEED_SCALAR: tuple[str, ...] = ("vbar", "arms", "keep_frac",
                                    "lam1", "lam2")
#: Per-neuron arrays that depend only on the weights.
PER_MLP: tuple[str, ...] = ("wn", "w4")


def _relu_moment_grads(m, s):
    """``(Phi, phi, mu0, dvh_dm, dvh_ds)`` for the linearised rectifier.

    ``E[relu] = m Phi + s phi`` and ``E[relu^2] = (m^2 + s^2) Phi + m s phi``
    give, after the ``phi'(a) = -a phi`` cancellations,

        d E[relu]/dm  = Phi          d E[relu]/ds  = phi
        d E[relu^2]/dm = 2 E[relu]   d E[relu^2]/ds = 2 s Phi

    so the post-ReLU VARIANCE derivatives follow by
    ``dV = dE[relu^2] - 2 E[relu] dE[relu]``.
    """
    a = m / s
    Ph = C.norm_cdf(a)
    ph = C.norm_pdf(a)
    mu0 = m * Ph + s * ph
    return Ph, ph, mu0, 2.0 * mu0 * (1.0 - Ph), 2.0 * (s * Ph - mu0 * ph)


def top_modes(x, mu, k: int = 2, iters: int = 8, seed: int = 12345):
    """Top-``k`` eigenvectors of ``Cov(h^32)``, by power iteration, no Gram.

    The Monte-Carlo error of ``mu_hat`` has covariance ``Cov(h^32)/N`` exactly,
    so the leading eigendirections of ``Cov(h^32)`` are *by construction* the
    directions in which the residual this head predicts is largest -- and a
    sibling measurement finds the deterministic closure's error at ``L = 32``
    concentrated in the same top direction (``q_1 = 3.4-7.1``).  Both arms
    therefore want the same feature.

    Forming the Gram costs ``2 N n^2 = 3.3e9`` FLOPs at ``N = 25,000``, 1.2% of
    the whole budget.  Power iteration never forms it: each sweep is
    ``x @ V`` then ``x.T @ (x V)``, ``4 N n k`` FLOPs, so eight sweeps at
    ``k = 2`` cost 4.1e8 -- 0.15% of the budget, and 8x less again if the
    sweeps are run on a subsample.

    The sign of each eigenvector is fixed by ``<u, mu> > 0``, which is a
    permutation-INVARIANT rule, so the per-neuron feature ``u_j`` transforms
    correctly under relabelling of the output neurons.
    """
    n = x.shape[1]
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((n, k)))[0].astype(np.float32)
    N = x.shape[0]
    muf = np.asarray(mu, dtype=np.float32)
    for _ in range(iters):
        Y = x @ V
        V = (x.T @ Y) / np.float32(N) - np.outer(muf, muf @ V)
        V = np.linalg.qr(V)[0]
    Y = x @ V
    lam = (np.sum(np.asarray(Y, dtype=np.float64) ** 2, axis=0) / N
           - (np.asarray(muf @ V, dtype=np.float64)) ** 2)
    V = np.asarray(V, dtype=np.float64)
    sgn = np.where(np.asarray(mu, dtype=np.float64) @ V >= 0.0, 1.0, -1.0)
    return V * sgn, lam


def transport(weights, ms, ss, l0, dm, dvz, wsq=None):
    """Carry a PRE-activation ``(mean, variance)`` gap at layer ``l0`` forward.

    ``ms``/``ss`` are the per-layer pre-activation mean and sd -- the state the
    linearisation is taken at.  Returns the induced perturbation of the final
    layer's ``E[relu(z^32)]``.

    The recursion is the exact first-order chain rule of the Gaussian
    rectifier moments,

        dmu_l  = Phi dm + phi ds
        dvh_l  = 2 mu0 (1 - Phi) dm + 2 (s Phi - mu0 phi) ds,   ds = dvz/(2 s)
        dm_{l+1}  = W' dmu_l                    (exact)
        dvz_{l+1} = (W .^ 2)' dvh_l             (diagonal only)

    Only the last line is an approximation -- it drops the off-diagonal
    covariance gaps -- and it costs efficiency, never bias: whatever the
    coefficients, the output is a LINEAR functional of inputs that are exactly
    mean zero, so the correction it defines is exactly mean zero too.  That is
    the whole reason to route everything through exactly-known moments.

    ``wsq`` caches ``W .^ 2`` across calls; it is a function of the weights
    alone, so a runtime implementation builds it once per MLP.
    """
    depth = len(weights)
    for l in range(l0, depth):
        s = ss[l]
        ds = dvz / (2.0 * s)
        Ph, ph, mu0, gvm, gvs = _relu_moment_grads(ms[l], s)
        if l == depth - 1:
            return Ph * dm + ph * ds
        dmu = Ph * dm + ph * ds
        dvh = gvm * dm + gvs * ds
        W = np.asarray(weights[l + 1], dtype=np.float64)
        dm = W.T @ dmu
        dvz = (W * W if wsq is None else wsq[l + 1]).T @ dvh
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# The instrumented scored pass
# ---------------------------------------------------------------------------
def extract(weights, seed: int, *, tau: float | None = 2.5,
            n_samples: int = 25_000, n_pilot: int = 225,
            kmax: int = 4, want_relu1: bool = True, want_q2: bool = True,
            gate_rows: int = GATE_ROWS) -> dict:
    """Run the shipped sparse pass and return the full per-neuron feature set.

    Identical to ``submission/estimator.py::_sparse`` on the scored row -- same
    generator, same pilot, same mask, same pre-sliced matmuls -- with every
    accumulator bolted on AFTER the quantity it reads, so nothing here can
    perturb ``mu``.
    """
    n = weights[0].shape[0]
    depth = len(weights)
    rng = np.random.default_rng(seed)

    # ---- pilot -----------------------------------------------------------
    x = rng.standard_normal((n_pilot, n), dtype=np.float32)
    alpha_p, mean_hp, s_p, m_p = [], [], [], []
    for w in weights:
        z = x @ w
        m = np.mean(z, axis=0)
        v = np.maximum(np.mean(z * z, axis=0) - m * m, VAR_FLOOR)
        sd = np.sqrt(v)
        alpha_p.append(m / sd)
        m_p.append(m.astype(np.float64))
        s_p.append(sd.astype(np.float64))
        x = np.maximum(z, 0.0)
        mean_hp.append(np.mean(x, axis=0))

    # ---- masks and pre-sliced weights ------------------------------------
    subs, biases, keeps = [], [], []
    keep_prev = None
    for l, w in enumerate(weights):
        keep = None if (tau is None or l == depth - 1) else (alpha_p[l] > -tau)
        keeps.append(keep)
        wr = w if keep_prev is None else w[keep_prev, :]
        subs.append(wr if keep is None else wr[:, keep])
        if keep_prev is None:
            biases.append(None)
        else:
            dead = mean_hp[l - 1] * (1.0 - keep_prev.astype(np.float32))
            wd = w if keep is None else w[:, keep]
            biases.append(dead @ wd)
        keep_prev = keep
    keep_frac = (1.0 if tau is None else
                 float(np.mean([np.mean(alpha_p[l] > -tau)
                                for l in range(depth - 1)])))

    # ---- scored pass -----------------------------------------------------
    x0 = rng.standard_normal((n_samples, n), dtype=np.float32)
    x = x0
    z1 = h1 = None
    gm, gv = [], []      # per-layer (m, v) from the first ``gate_rows`` rows
    for l in range(depth):
        z = x @ subs[l]
        if biases[l] is not None:
            z = z + biases[l]
        if l == 0:
            z1 = z
        # The pruned columns are absent from ``z``, so the scored-pass state is
        # SCATTERED into a full-width copy of the pilot's -- a dead neuron's
        # gate is Phi(alpha) at alpha < -tau, i.e. under 0.6%, and the pilot
        # is a perfectly good estimate of a number that small.
        zg = z[:gate_rows]
        mg = np.mean(zg, axis=0).astype(np.float64)
        vg = np.maximum(np.mean(zg * zg, axis=0).astype(np.float64) - mg * mg,
                        VAR_FLOOR)
        if keeps[l] is None:
            gm.append(mg)
            gv.append(vg)
        else:
            fm = m_p[l].copy()
            fv = (s_p[l] * s_p[l]).copy()
            fm[keeps[l]] = mg
            fv[keeps[l]] = vg
            gm.append(fm)
            gv.append(fv)
        if l == 1:
            # Full-sample first and second moments of ``z^2``.  ``E[z^2]`` and
            # ``Cov(z^2)`` are EXACTLY known (``z^1`` is exactly Gaussian, so
            # ``Cov(relu(z^1))`` is the arc-cosine kernel and ``Cov(z^2) = W^2'
            # Cov(relu z^1) W^2``), which makes both gaps below exactly mean
            # zero -- and the second one is the FULL layer-1 covariance gap
            # already contracted onto the direction that needs it, at the cost
            # of one elementwise square of the (N, |keep|) array.
            z2m = np.mean(z, axis=0).astype(np.float64)
            z2q = np.mean(z * z, axis=0).astype(np.float64)
        x = np.maximum(z, 0.0)
        if l == 0:
            h1 = x
    mu = np.mean(x, axis=0).astype(np.float64)

    # ---- layer-1 Hermite control variates --------------------------------
    cvs = C.hermite_cv(x0, z1, x, weights[0], kmax=kmax, split=True)
    while len(cvs) < 3:
        cvs.append(np.zeros(n, dtype=np.float64))

    # ---- final-layer sample moments --------------------------------------
    # SHAPE features only -- ``s``, ``alpha``, ``sd_mc``, the two standardised
    # cumulants -- none of which is a correction, so none of them needs the
    # full sample.  ``gate_rows`` of it estimates ``s`` to 0.8% and the whole
    # block costs five passes over an (n, 256) array instead of over a
    # (25000, 256) one.  ``mu`` above is untouched and is the estimate.
    xg, zg = x[:gate_rows], z[:gate_rows]
    mug = np.mean(xg, axis=0).astype(np.float64)
    ex2 = np.mean(xg * xg, axis=0).astype(np.float64)
    vh = np.maximum(ex2 - mug * mug, 0.0)
    m32 = np.mean(zg, axis=0).astype(np.float64)
    d32 = zg - m32.astype(zg.dtype)
    d2 = d32 * d32
    v32 = np.maximum(np.mean(d2, axis=0).astype(np.float64), VAR_FLOOR)
    s32 = np.sqrt(v32)
    gam1 = np.mean(d2 * d32, axis=0).astype(np.float64) / (v32 * s32)
    gam2 = np.mean(d2 * d2, axis=0).astype(np.float64) / (v32 * v32) - 3.0
    a32 = m32 / s32
    Ph32 = C.norm_cdf(a32)
    ph32 = C.norm_pdf(a32)

    # ---- exactly-mean-zero layer-1 gaps, and their transport --------------
    W1 = np.asarray(weights[0], dtype=np.float64)
    sig1 = np.sqrt(np.maximum(np.sum(W1 * W1, axis=0), VAR_FLOOR))
    h1m = np.mean(h1, axis=0).astype(np.float64)
    h1sq = np.mean(h1 * h1, axis=0).astype(np.float64)
    mu0_1 = sig1 * INV_SQRT_2PI
    dmu1 = h1m - mu0_1                          # exact: E relu(z1) = sig/sqrt(2pi)
    dvh1 = (h1sq - 0.5 * sig1 * sig1) - 2.0 * mu0_1 * dmu1  # exact: E relu^2 = sig^2/2

    ms_p = [np.asarray(v, dtype=np.float64) for v in m_p]
    ss_p = [np.asarray(v, dtype=np.float64) for v in s_p]
    ms_g = [np.asarray(v, dtype=np.float64) for v in gm]
    ss_g = [np.sqrt(v) for v in gv]

    # The exact layer-1/2 moments.  Needed by ``mfv2`` unconditionally, so the
    # ``relu1``/``q2`` blocks below get them for free.
    mh1, Ch1, m2, C2 = C.layer12_moments(weights)
    W2 = np.asarray(weights[1], dtype=np.float64)
    zero = np.zeros(n)
    dm2 = W2.T @ dmu1                       # exact transport of the mean gap
    dvz2_diag = (W2 * W2).T @ dvh1          # diagonal-only approximation
    k2 = keeps[1]
    dvz2 = np.zeros(n)
    c2d = np.diag(C2)
    if k2 is None:
        dvz2 = z2q - 2.0 * m2 * z2m + m2 * m2 - c2d
    else:
        dvz2[k2] = (z2q - 2.0 * m2[k2] * z2m + m2[k2] * m2[k2] - c2d[k2])
    wsq = None
    mfm = transport(weights, ms_p, ss_p, 1, dm2, zero, wsq)
    mfv = transport(weights, ms_p, ss_p, 1, zero, dvz2_diag, wsq)
    mfv2 = transport(weights, ms_p, ss_p, 1, zero, dvz2, wsq)
    mfmg = transport(weights, ms_g, ss_g, 1, dm2, zero, wsq)
    mfvg = transport(weights, ms_g, ss_g, 1, zero, dvz2_diag, wsq)
    mfv2g = transport(weights, ms_g, ss_g, 1, zero, dvz2, wsq)

    # The shipped ``cv1mf``: the same mean channel with the rectifier Jacobian
    # frozen at ``Phi(alpha_pilot)`` and no ``phi ds`` term at all.  Kept
    # verbatim so the new channels are measured AGAINST it, not instead of it.
    gates_p = [C.norm_cdf(np.asarray(a_, dtype=np.float64)) for a_ in alpha_p]
    gates_g = [C.norm_cdf(m_ / s_) for m_, s_ in zip(ms_g, ss_g)]
    out = {}
    for nm, gates in (("cv1mf", gates_p), ("cv1mfg", gates_g)):
        prop = dmu1
        for l in range(1, depth):
            prop = prop @ np.asarray(weights[l], dtype=np.float64)
            if l < depth - 1:
                prop = prop * gates[l]
        out[nm] = prop * Ph32

    # ---- the exactly-integrable blocks -----------------------------------
    out["relu1"] = (C.relu1_cv(h1, x, mh1, Ch1, split=True) if want_relu1
                    else np.zeros(n))
    if want_q2:
        A2 = C.kink_frame(weights, ms_p, [np.diag(s * s) for s in ss_p], Q2_K)
        z2 = np.maximum(z1, 0.0) @ np.asarray(weights[1], dtype=np.float64)
        out["q2"] = C.quad2_cv(z2, x, m2, C2, A2, split=True)
    else:
        out["q2"] = np.zeros(n)

    U, lam = top_modes(x, mu, k=2, iters=8)

    wl = np.asarray(weights[-1], dtype=np.float64)
    wn = np.sum(wl * wl, axis=0)
    out.update({
        "mu": mu,
        "cv1": cvs[0], "cv2": cvs[1], "cv3": cvs[2],
        "mfm": mfm, "mfv": mfv, "mfv2": mfv2,
        "mfmg": mfmg, "mfvg": mfvg, "mfv2g": mfv2g,
        "s": s32, "Phi": Ph32, "phi": ph32, "alpha": a32,
        "sd_mc": np.sqrt(vh / n_samples), "gam1": gam1, "gam2": gam2,
        "dpilot": mu - np.asarray(mean_hp[-1], dtype=np.float64),
        "alpha_p": np.asarray(alpha_p[-1], dtype=np.float64),
        "u1": U[:, 0], "u2": U[:, 1],
        "lam1": float(lam[0]), "lam2": float(lam[1]),
        "wn": wn,
        "w4": n * np.sum(wl ** 4, axis=0) / np.maximum(wn * wn, 1e-30),
        "vbar": float(np.mean(vh)),
        "arms": float(np.sqrt(np.mean(a32 * a32))),
        "keep_frac": keep_frac,
    })
    return {k: (v if np.isscalar(v) else np.asarray(v, dtype=np.float64))
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def reference(weights, n_samples: int, seed: int, *, chunk: int = 2048,
              use_cv: bool = True) -> np.ndarray:
    """Dense float64-accumulated ``E[relu(z^32)]`` with the k=1 CV applied.

    The input-linear control variate ``correction_j = (1/N) sum_s (x_s . xbar)
    y_sj - (xbar . xbar) mu_j`` is exactly mean zero (``E[x] = 0``), so the
    reference stays unbiased; it is estimated in SPLIT-SAMPLE form (``xbar``
    from one half against the cross-moment of the other, both ways) so the
    ``Cov(|x|^2, y)/N`` self-term of the one-pass form is absent.

    Everything it needs is one extra ``(width, width)`` cross-moment
    accumulation per half, i.e. ``2 N width^2`` FLOPs against the pass's
    ``2 N depth width^2`` -- 3% for a measured variance reduction, which is a
    pure multiplier on the effective reference sample count.
    """
    n = weights[0].shape[0]
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    acc = []
    for lo, hi in ((0, half), (half, n_samples)):
        cnt = hi - lo
        ysum = np.zeros(n, dtype=np.float64)
        xsum = np.zeros(n, dtype=np.float64)
        cross = np.zeros((n, n), dtype=np.float64) if use_cv else None
        done = 0
        while done < cnt:
            nb = min(chunk, cnt - done)
            x0 = rng.standard_normal((nb, n), dtype=np.float32)
            h = x0
            for w in weights:
                h = np.maximum(h @ w, 0.0)
            ysum += h.sum(axis=0, dtype=np.float64)
            if use_cv:
                xsum += x0.sum(axis=0, dtype=np.float64)
                cross += np.asarray(x0.T @ h, dtype=np.float64)
            done += nb
        acc.append((ysum / cnt, xsum / cnt, None if cross is None
                    else cross / cnt))
    (mu_a, xb_a, cr_a), (mu_b, xb_b, cr_b) = acc
    mu = 0.5 * (mu_a + mu_b)
    if not use_cv:
        return mu
    corr = 0.5 * ((cr_a.T @ xb_b - float(xb_a @ xb_b) * mu_a)
                  + (cr_b.T @ xb_a - float(xb_b @ xb_a) * mu_b))
    return mu - corr


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
def ridge_solve(G, b, lam, nrow):
    Gr = G.copy()
    Gr[np.diag_indices(len(Gr))] += lam * nrow
    return np.linalg.solve(Gr, b)


class RFRidge:
    """Random-feature ridge: ``y ~ [x, tanh(x A + c)] beta``, closed form.

    The residual after the control variates is a nearly-linear function of them
    with coefficients that depend on the per-neuron regime, so the capacity
    that can matter is exactly an interaction between the correction columns
    and the shape columns.  A random tanh layer spans those interactions
    without an optimiser, without a learning rate and without a training loop
    that can silently fail to converge -- which is the whole reason to try it
    before an MLP.  ``linear=True`` keeps the raw columns alongside, so the
    model can never do worse than ridge at the same penalty.

    **Why it does not scale past ~2,000 features here, and SGD does.**  The
    normal equations cost ``n_rows * n_out^2``; at 800,000 rows that is 3.4e12
    FLOPs at 2,048 features and 1.4e13 at 4,096.  A gradient step costs
    ``n_rows * n_params`` per epoch, i.e. ``n_out``-times less, so past a couple
    of thousand features :class:`SGDHead` is the only affordable estimator on
    this hardware.  Both are reported, and the crossover is measured rather
    than assumed.

    ``fit`` streams: the design matrix at 1e6 rows x 2,048 features is 16 GB in
    float64 and is never materialised.
    """

    def __init__(self, n_in: int, n_feat: int, seed: int = 0,
                 scale: float = 1.0, linear: bool = True, n_lin: int = 0):
        rng = np.random.default_rng(seed)
        n_nl = n_in - n_lin
        self.A = (rng.standard_normal((n_nl, n_feat))
                  * (scale / math.sqrt(n_nl))).astype(np.float32)
        self.c = rng.uniform(-math.pi, math.pi, n_feat).astype(np.float32)
        self.linear = linear
        self.n_in, self.n_feat, self.n_lin = n_in, n_feat, n_lin
        self.n_out = n_feat + (n_in if linear else 0)

    def phi(self, Xs):
        """First ``n_lin`` columns pass through; the rest feed the tanh layer.

        The split is what makes the comparison against plain ridge **nested**:
        the linear block is the full rich design, so the random features can
        only ever be an addition to it and any loss is the penalty's doing
        rather than a different hypothesis class.
        """
        X = np.asarray(Xs, dtype=np.float32)
        h = np.tanh(X[:, self.n_lin:] @ self.A + self.c)
        return np.concatenate([X, h], axis=1) if self.linear else h

    def n_params(self) -> int:
        return self.A.size + self.c.size + self.n_out

    def fit(self, X, y, lams, block: int = 16384, row_scale=None) -> dict:
        """Return ``{lam: beta}`` from streamed normal equations.

        ``row_scale`` multiplies every basis row, which is how the SCALE-FREE
        parameterisation is reconciled with the METRIC.  The head is written as
        ``y_j = sd_j f(z_j)`` because that is the form in which the map is a
        bounded function of bounded inputs -- but least squares on ``y_j/sd_j``
        minimises ``sum ((y-p)/sd)^2``, weighting each neuron by ``1/sd^2``,
        while the score is the UNWEIGHTED ``sum (y-p)^2``.  Passing
        ``row_scale = sd`` fits ``sd f(z)`` against ``y`` directly, which is
        the same hypothesis class under the right loss.
        """
        G = np.zeros((self.n_out, self.n_out), dtype=np.float64)
        b = np.zeros(self.n_out, dtype=np.float64)
        for lo in range(0, len(X), block):
            P = self.phi(X[lo:lo + block])
            if row_scale is not None:
                P = P * np.asarray(row_scale[lo:lo + block],
                                   dtype=np.float32)[:, None]
            G += (P.T @ P).astype(np.float64)
            b += P.T @ np.asarray(y[lo:lo + block], dtype=np.float32)
        # Penalise in the COLUMN-SCALED basis, exactly as the plain ridge
        # does.  Without it the random features and the row-scaled linear
        # block sit at wildly different magnitudes -- the linear columns carry
        # a factor ``sd ~ 1e-3`` -- and no single ``lam`` can shrink both
        # sensibly, which shows up as the grid pinning at its own endpoint and
        # the nesting against plain ridge silently failing.
        n = len(X)
        dg = np.sqrt(np.maximum(np.diag(G) / n, 1e-300))
        dg = np.where(dg > 0, dg, 1.0)
        Gs = G / np.outer(dg, dg)
        bs = b / dg
        return {lam: ridge_solve(Gs, bs, lam, n) / dg for lam in lams}

    def predict(self, X, beta, block: int = 16384, row_scale=None):
        out = np.empty(len(X), dtype=np.float64)
        bf = np.asarray(beta, dtype=np.float32)
        for lo in range(0, len(X), block):
            P = self.phi(X[lo:lo + block])
            if row_scale is not None:
                P = P * np.asarray(row_scale[lo:lo + block],
                                   dtype=np.float32)[:, None]
            out[lo:lo + block] = P @ bf
        return out


class SGDHead:
    """Per-neuron MLP head: ``n_in -> h... -> 1``, tanh, Adam, minibatch numpy.

    Weight-shared across output neurons by construction -- one row per neuron,
    the same parameters for every one -- so the estimator is **exactly
    equivariant to permutation of the hidden units of the target network** and
    generalises to fresh MLPs by the same argument that lets 256 neurons of one
    MLP be 256 training rows.
    """

    def __init__(self, n_in: int, hidden=(64,), seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = (n_in, *hidden, 1)
        self.W = [rng.standard_normal((dims[i], dims[i + 1])).astype(np.float32)
                  / math.sqrt(dims[i]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(d, dtype=np.float32) for d in dims[1:]]

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(v.size for v in self.b)

    def forward(self, X):
        h = np.asarray(X, dtype=np.float32)
        acts = [h]
        for i, (w, bb) in enumerate(zip(self.W, self.b)):
            h = h @ w + bb
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        return acts

    def predict(self, X, block: int = 65536, out_scale=None) -> np.ndarray:
        out = np.empty(len(X), dtype=np.float64)
        for lo in range(0, len(X), block):
            out[lo:lo + block] = self.forward(X[lo:lo + block])[-1][:, 0]
        return out if out_scale is None else out * np.asarray(out_scale)

    def fit(self, X, y, *, epochs: int = 30, lr: float = 3e-3,
            batch: int = 8192, wd: float = 0.0, seed: int = 0,
            val=None, verbose: bool = False, out_scale=None):
        rng = np.random.default_rng(seed + 7)
        ps = self.W + self.b
        ms = [np.zeros_like(p) for p in ps]
        vs = [np.zeros_like(p) for p in ps]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        y = np.asarray(y, dtype=np.float32)
        nl = len(self.W)
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for lo in range(0, len(X), batch):
                sel = idx[lo:lo + batch]
                acts = self.forward(X[sel])
                m = len(sel)
                os_ = (1.0 if out_scale is None
                       else np.asarray(out_scale[sel], dtype=np.float32))
                g = (2.0 / m) * ((acts[-1][:, 0] * os_ - y[sel]) * os_
                                 )[:, None]
                gW, gb = [None] * nl, [None] * nl
                for i in range(nl - 1, -1, -1):
                    gW[i] = acts[i].T @ g
                    gb[i] = g.sum(0)
                    if i:
                        g = (g @ self.W[i].T) * (1.0 - acts[i] * acts[i])
                grads = [gW[i] + wd * self.W[i] for i in range(nl)] + gb
                step += 1
                bc1 = 1 - b1 ** step
                bc2 = 1 - b2 ** step
                for p, gr, mm, vv in zip(ps, grads, ms, vs):
                    mm *= b1
                    mm += (1 - b1) * gr
                    vv *= b2
                    vv += (1 - b2) * gr * gr
                    p -= lr * (mm / bc1) / (np.sqrt(vv / bc2) + eps)
            if verbose and val is not None and (ep + 1) % 5 == 0:
                print(f"      epoch {ep + 1:3d}  val {val(self):.6e}",
                      flush=True)
        return self
