"""Offline-trained residual corrector: features, ridge/MLP heads, and the fit.

Why this family
---------------
The low-order barrier (``docs/floor_theorem.md``) closes every surrogate whose
ANOVA content is concentrated at order <= 1 at ``1/(1 - f_1) = 1.38x``.  A map
*fitted offline to the weight -> mean relation* is not built from low-order
structure, and offline training + precomputation load at **zero FLOPs** at
grade time (``fnp.load``, measured 0).  This module is that map.

The construction
----------------
The shipped estimator is sparse Monte Carlo (``docs/sparse_sign_stable.md``),
whose final-layer prediction is ``mu_hat_j = mean_s relu(z^32_{s,j})``.  We
predict the **residual** ``target_j = mu_true_j - mu_hat_j`` with a linear head
over predict-time features and add the prediction back.  Everything the head
sees costs ~0.3% of the scored pass, and the head itself is a matvec against a
``(n_features,)`` vector loaded from disk.

The features that carry the signal are **layer-1 Hermite control variates**.
``z^1_i = x . W^1[:,i]`` is *exactly* Gaussian with known scale
``sigma_i = ||W^1[:,i]||``, so for ``t_i = z^1_i / sigma_i``

    E[He_k(t_i)] = 0    exactly, for every k >= 1                        (*)
    Cov(He_k(t_i), He_l(t_j)) = delta_kl k! rho_ij^k     (Mehler)        (**)

with ``rho`` the correlation matrix of the layer-1 pre-activations.  (*) makes
the sample means of ``He_k(t)`` free, exactly-mean-zero control variates; (**)
makes their Gram *analytic and block diagonal in k*, so the optimal
coefficients need no estimated covariance matrix.  Writing ``u_k = G_k^{-1}
d_k`` for the sample-mean vector ``d_k`` and ``w_s = sum_k u_k . (g_{k,s} -
d_k)``, the whole correction collapses to two length-N matvecs:

    correction_j = (1/N) sum_s w_s y_{sj}

so it costs ``O(N * width)``, not ``O(N * width^2)``.

Measured population shares of ``Var(relu(z^32_j))`` explained (4 local MLPs,
400k samples, ``scripts/28 --mode anova``): ``k=1`` 23-29%, ``k<=2`` 38-48%,
``k<=3`` +2%, ``k<=6`` +5%.  The ``k=1`` block is *identical* to the optimal
input-linear control variate (the change of basis ``u_1 = rho^{-1} d_1``
cancels ``W^1`` exactly), which is why it lands on the barrier's 1.38x.  The
``k=2`` block is what escapes it: ``He_2(w_i . x)`` is an order-2 object along
the network's own first-layer directions, and the barrier only ever *computed*
the ``k=1`` instance.

Each extra block costs ``p/N = 256/8500 = 3.0%`` of the residual in
overfitting, so ``k <= 2`` is the optimum and ``k >= 3`` is not worth its own
noise.  The offline head recovers part of that 3% by learning a shrinkage on
each block -- a fitted James-Stein weight rather than an assumed one.

The design was 28 columns and is now 15; see :data:`DROPPED`.  Everything
removed measured at exactly 1.000x in the leave-one-group-out table and three
of the four groups cost passes over the ``(N, width)`` sample array, which is
participant wall time billed at ``lambda = 1e11`` FLOP/s.  The full 28-column
design survives as :data:`FEATURES_FULL` so those ablations stay reproducible.

What is NOT here, and was measured first: **Stein control variates built from
the network's own gradient** (``docs/stein_cv.md``, ``scripts/30_stein_cv.py``).
``h = c.grad psi - (c.x) psi`` is exactly mean-zero for any Lipschitz ``psi``
-- verified to Monte-Carlo error -- but the whole family reaches R^2 = 11.5%
against ``relu(z^32_j)`` and adds **+0.4 points** on top of the layer-1
Hermite family, against a pre-registered bar of R^2 > 0.75.

Nothing here is imported by ``submission/estimator.py``: the submission carries
a flopscope-only copy, and ``tests/test_corrector_parity.py`` asserts the two
agree bitwise.
"""

from __future__ import annotations

import math

import numpy as np

#: 1 / sqrt(2 pi).
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

#: Floor applied to pre-activation variances before a square root.
VAR_FLOOR = 1e-12

#: Relative jitter on the analytic Hermite Gram's diagonal.  Insurance only:
#: measured ``cond(2 rho .^ 2) = 2.41``, because squaring O(1/16) correlations
#: makes them O(1/256).  ``rho`` itself -- the k=1 Gram -- has ``cond = 3.8e8``
#: at this shape, which is exactly why the k=1 block is evaluated in the input
#: basis where the Gram is the identity (see :func:`hermite_cv`).
GRAM_JITTER = 1e-6


def norm_cdf(x):
    """``Phi``, bit-identical to ``flopscope.stats.norm.cdf``."""
    from flopscope.stats._erf import _erf  # noqa: PLC0415
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def norm_pdf(x):
    """``phi``, bit-identical to ``flopscope.stats.norm.pdf``."""
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


# ---------------------------------------------------------------------------
# Feature block.  Order is FROZEN: the shipped coefficient vector indexes it.
# ---------------------------------------------------------------------------
#: The full RESEARCH design.  Every group ablation in ``scripts/28 --mode fit``
#: indexes this tuple, so it must not be reordered or the published tables
#: stop meaning what they say.
FEATURES_FULL: tuple[str, ...] = (
    "one",          # intercept
    "cv1",          # layer-1 Hermite k=1 control variate (== input-linear CV)
    "cv1_Phi",
    "cv1_a",
    "cv2",          # layer-1 Hermite k=2 control variate
    "cv2_Phi",
    "cv2_a",
    "cv3",          # layer-1 Hermite k=3 control variate
    "cv1mf",        # layer-1 mean gap pushed through the mean-field Jacobian
    "cv1mf_Phi",
    "cv1mf_a",
    "gap",          # Rao-Blackwell gap  g - mu_hat,  g = m Phi + s phi
    "gap_Phi",
    "gap_a",
    "sk",           # Edgeworth skew term
    "ku",           # Edgeworth kurtosis term
    "mu",           # the estimate itself (a global multiplicative shrink)
    "mu_Phi",
    "s",            # sample sd of z^32
    "Phi",
    "phi",
    "a",            # alpha = m / s
    "sd_mc",        # sqrt(Var(relu z) / N): the per-neuron MC noise scale
    "dpilot",       # mu_hat - pilot mean  (a second, weak estimate)
    "wn",           # ||W^32[:,j]||^2 - 1
    "w4",           # width * sum_i W^32[i,j]^4 / ||.||^4 - 3
    "vbar",         # per-MLP mean of Var(relu z) - 0.05
    "arms",         # per-MLP rms|alpha| - 3.4
)

#: Columns the SHIPPED kernel does not compute at all.
#:
#: Every one of them measures at **exactly 1.000x** in the leave-one-group-out
#: table of ``docs/learned_corrector.md`` sec 5 -- removing the group leaves
#: the validation gain at 1.504x, unchanged to three decimals -- while three of
#: the four groups cost real passes over the ``(N, width)`` sample array, which
#: is participant wall time billed at ``lambda = 1e11`` FLOP/s:
#:
#:   * ``gap``/``sk``/``ku`` need ``gamma_1`` and ``gamma_2``, i.e. ``d^3`` and
#:     ``d^4`` over the whole final-layer sample: five passes;
#:   * ``sd_mc`` and ``vbar`` both need ``vh = Var(relu z^32)``, i.e.
#:     ``mean(x*x)``: two more passes;
#:   * ``wn``/``w4`` are ``(width, width)`` reductions, cheap but worthless;
#:   * ``mu``/``mu_Phi`` (the multiplicative shrink) and ``cv3`` (zero at
#:     ``CV_KMAX = 2``) are free and worthless.
#:
#: Dropping them removes **eight** passes over the ``(8500, 256)`` array.  The
#: fitted head is re-fitted on the reduced design, not merely masked, so the
#: ridge shrinkage is the right one for the columns that remain.
DROPPED: tuple[str, ...] = (
    "cv3",
    "gap", "gap_Phi", "gap_a", "sk", "ku",
    "mu", "mu_Phi",
    "sd_mc",
    "wn", "w4", "vbar", "arms",
)

#: What the submission actually builds and what the shipped ``beta`` indexes.
FEATURES: tuple[str, ...] = tuple(f for f in FEATURES_FULL if f not in DROPPED)

N_FEATURES = len(FEATURES)
N_FEATURES_FULL = len(FEATURES_FULL)

#: Primitive arrays :func:`sparse_mc_features` returns per output neuron.
PRIMITIVES: tuple[str, ...] = (
    "mu", "cv1", "cv2", "cv3", "cv1mf", "gap", "sk", "ku", "alpha", "Phi",
    "phi", "s", "vh", "sd_mc", "dpilot", "wn", "w4", "gam1", "gam2",
)
#: Per-MLP scalars.
SCALARS: tuple[str, ...] = ("vbar", "arms", "keep_frac")


def feature_columns(f) -> dict:
    """Every named column, from the primitives.

    Single source of truth for the derived columns, so the research path and
    the shipped path cannot drift.  Building all of them is free here (this is
    the offline numpy path); the shipped kernel builds only ``FEATURES``.
    """
    a, Ph = f["alpha"], f["Phi"]
    one = np.ones_like(a)
    cv1, cv2, gap, mf, mu = f["cv1"], f["cv2"], f["gap"], f["cv1mf"], f["mu"]
    return {
        "one": one,
        "cv1": cv1, "cv1_Phi": cv1 * Ph, "cv1_a": cv1 * a,
        "cv2": cv2, "cv2_Phi": cv2 * Ph, "cv2_a": cv2 * a,
        "cv3": f["cv3"],
        "cv1mf": mf, "cv1mf_Phi": mf * Ph, "cv1mf_a": mf * a,
        "gap": gap, "gap_Phi": gap * Ph, "gap_a": gap * a,
        "sk": f["sk"], "ku": f["ku"],
        "mu": mu, "mu_Phi": mu * Ph,
        "s": f["s"], "Phi": Ph, "phi": f["phi"], "a": a,
        "sd_mc": f["sd_mc"], "dpilot": f["dpilot"],
        "wn": f["wn"] - 1.0, "w4": f["w4"] - 3.0,
        "vbar": one * (float(f["vbar"]) - 0.05),
        "arms": one * (float(f["arms"]) - 3.4),
    }


def build_design(f, names: tuple[str, ...] | None = None) -> np.ndarray:
    """``(width, len(names))`` design matrix; ``names`` defaults to what ships."""
    cols = feature_columns(f)
    return np.stack([cols[k] for k in (names or FEATURES)], axis=1)


# ---------------------------------------------------------------------------
# The layer-1 Hermite control variate
# ---------------------------------------------------------------------------
def hermite_cv(x, z1, y, w1, kmax: int = 3, split: bool = True):
    """Layer-1 Hermite control-variate corrections, one per output neuron.

    ``x`` is the (N, width) input draw, ``z1 = x @ W^1`` the layer-1
    pre-activation, ``y`` the final-layer activations and ``w1 = W^1``.
    Returns a list ``[c_1, ..., c_kmax]`` of (width,) arrays; subtracting
    ``c_k`` from ``mu_hat`` removes the part of the sampling error the order-k
    block explains.

    The k=1 block is evaluated in the *input* basis, where the Gram is exactly
    the identity: ``u_1 = rho^{-1} d_1`` unwinds to ``W^{1,-1} xbar``, so
    ``w_s = (x_s - xbar) . xbar`` -- no solve, no conditioning problem, and
    the k=1 block is thereby *proved* to be the optimal input-linear control
    variate rather than merely resembling one.

    ``split`` picks the estimator of the coefficient vector.  The one-pass
    form ``dbar' G^-1 chat_j`` reuses the same samples for ``dbar`` and for
    the covariance ``chat_j``, which leaves the self-term

        E[dbar' G^-1 chat_j] = Cov(g' G^-1 g, y_j) / N

    -- a REAL bias, not noise, and it grows with k because ``He_k(t)^2`` has
    an increasingly heavy even-Hermite content that couples to ``y``.
    Measured at k=3 it reverses the sign of the correction (0.84x instead of
    a gain).  ``split=True`` uses two halves, taking ``dbar`` from one and
    ``chat`` from the other in both directions, which is exactly unbiased at
    identical cost.
    """
    n = x.shape[0]
    sig1 = np.sqrt(np.maximum(np.sum(w1 * w1, axis=0), VAR_FLOOR))
    inv1 = (1.0 / sig1).astype(z1.dtype)
    # Normalising the columns first makes ``rho`` a Gram, hence exactly
    # symmetric (flopscope keeps the tag, and the divide that would lose it
    # never happens) with an exact unit diagonal.
    rho = None

    # (basis, Gram, per-column scale folded into d and u, constant offset)
    bases = [(x, None, None, 0.0)]
    if kmax >= 2:
        wn = w1 * inv1
        rho = wn.T @ wn
        G2 = (2.0 * rho) * rho
        np.fill_diagonal(G2, np.asarray(2.0 * (1.0 + GRAM_JITTER),
                                        dtype=G2.dtype))
        bases.append((z1 * z1, G2, inv1 * inv1, -1.0))
    if kmax >= 3:
        t = z1 * inv1
        G3 = ((6.0 * rho) * rho) * rho
        np.fill_diagonal(G3, np.asarray(6.0 * (1.0 + GRAM_JITTER),
                                        dtype=G3.dtype))
        bases.append((t * t * t - 3.0 * t, G3, None, 0.0))

    out = []
    for g, G, sc, off in bases:
        def _u(dd):
            dd = dd if sc is None else dd * sc
            dd = dd if off == 0.0 else dd + off
            return dd if G is None else np.linalg.solve(G, dd)

        def _w(gg, uu):
            v = gg @ (uu if sc is None else uu * sc)
            return v - np.mean(v)

        if split:
            h = n // 2
            g1, g2 = g[:h], g[h:]
            u1 = _u(np.mean(g1, axis=0))
            u2 = _u(np.mean(g2, axis=0))
            c = 0.5 * ((y[:h].T @ _w(g1, u2)) / h
                       + (y[h:].T @ _w(g2, u1)) / (n - h))
        else:
            c = (y.T @ _w(g, _u(np.mean(g, axis=0)))) / n
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# The sparse Monte-Carlo pass, instrumented.  Numpy research copy.
# ---------------------------------------------------------------------------
def sparse_mc_features(weights, seed: int, *, tau: float | None = 2.5,
                       n_samples: int = 8500, n_pilot: int = 150,
                       kmax: int = 3, split: bool = True):
    """Run the shipped sparse pass and return ``(mu_hat_final, primitives)``.

    Identical to ``whestfloor.kernels._sparse_mc`` on the scored row -- same
    generator, same pilot, same mask, same sliced matmuls -- with the feature
    accumulators bolted on *after* the quantities they read, so they cannot
    perturb the estimate.
    """
    n = weights[0].shape[0]
    depth = len(weights)
    rng = np.random.default_rng(seed)

    # ---- pilot ---------------------------------------------------------
    x = rng.standard_normal((n_pilot, n), dtype=np.float32)
    alpha, mean_h = [], []
    for w in weights:
        z = x @ w
        m = np.mean(z, axis=0)
        v = np.maximum(np.mean(z * z, axis=0) - m * m, VAR_FLOOR)
        alpha.append(m / np.sqrt(v))
        x = np.maximum(z, 0.0)
        mean_h.append(np.mean(x, axis=0))

    # ---- masks and pre-sliced weights ----------------------------------
    subs, biases = [], []
    keep_prev = None
    for l, w in enumerate(weights):
        keep = None if (tau is None or l == depth - 1) else (alpha[l] > -tau)
        wr = w if keep_prev is None else w[keep_prev, :]
        subs.append(wr if keep is None else wr[:, keep])
        if keep_prev is None:
            biases.append(None)
        else:
            dead = mean_h[l - 1] * (1.0 - keep_prev.astype(np.float32))
            wd = w if keep is None else w[:, keep]
            biases.append(dead @ wd)
        keep_prev = keep
    keep_frac = (1.0 if tau is None else
                 float(np.mean([np.mean(alpha[l] > -tau)
                                for l in range(depth - 1)])))

    # ---- scored pass ---------------------------------------------------
    x0 = rng.standard_normal((n_samples, n), dtype=np.float32)
    x = x0
    z1 = h1m = None
    for l in range(depth):
        z = x @ subs[l]
        if biases[l] is not None:
            z = z + biases[l]
        if l == 0:
            z1 = z
        x = np.maximum(z, 0.0)
        if l == 0:
            h1m = np.mean(x, axis=0)
    mu = np.mean(x, axis=0)

    cvs = hermite_cv(x0, z1, x, weights[0], kmax=kmax, split=split)
    while len(cvs) < 3:
        cvs.append(np.zeros(n, dtype=np.float32))

    # ---- final-layer sample moments ------------------------------------
    ex2 = np.mean(x * x, axis=0)
    vh = np.maximum(ex2 - mu * mu, 0.0)
    m = np.mean(z, axis=0)
    d = z - m
    v = np.maximum(np.mean(d * d, axis=0), VAR_FLOOR)
    s = np.sqrt(v)
    gam1 = np.mean(d ** 3, axis=0) / (v * s)
    gam2 = np.mean(d ** 4, axis=0) / (v * v) - 3.0
    a = m / s
    Ph = norm_cdf(a).astype(np.float32)
    ph = norm_pdf(a).astype(np.float32)
    g = m * Ph + s * ph
    sk = -(gam1 / 6.0) * s * a * ph
    ku = (gam2 / 24.0) * s * (a * a - 1.0) * ph

    # ---- mean-field propagation of the layer-1 mean gap (comparison arm)
    sig1 = np.sqrt(np.sum(weights[0] * weights[0], axis=0))
    d1 = h1m - sig1 * np.float32(INV_SQRT_2PI)
    gates = [norm_cdf(alpha[l]).astype(np.float32) for l in range(depth)]
    prop = d1
    for l in range(1, depth):
        prop = prop @ weights[l]
        if l < depth - 1:
            prop = prop * gates[l]
    cv1mf = prop * Ph

    wl = weights[-1]
    wn = np.sum(wl * wl, axis=0)
    prim = {
        "mu": mu, "cv1": cvs[0], "cv2": cvs[1], "cv3": cvs[2],
        "cv1mf": cv1mf, "gap": g - mu, "sk": sk, "ku": ku, "alpha": a,
        "Phi": Ph, "phi": ph, "s": s, "vh": vh,
        "sd_mc": np.sqrt(vh / n_samples), "dpilot": mu - mean_h[-1],
        "wn": wn, "w4": n * np.sum(wl ** 4, axis=0) / np.maximum(wn * wn, 1e-30),
        "gam1": gam1, "gam2": gam2,
    }
    prim = {k: np.asarray(v, dtype=np.float64) for k, v in prim.items()}
    prim["vbar"] = float(np.mean(vh))
    prim["arms"] = float(np.sqrt(np.mean(a * a)))
    prim["keep_frac"] = keep_frac
    prim["n_samples"] = n_samples
    return prim["mu"], prim


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
def design_scale(X: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.mean(X * X, axis=0))
    return np.where(s > 0, s, 1.0)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float,
              scale: np.ndarray | None = None) -> np.ndarray:
    """Closed-form ridge with per-column scaling.  No optimiser, no sklearn.

    The penalty is applied in the scaled basis so ``lam`` means the same thing
    for every feature; the returned coefficients are in the ORIGINAL basis so
    the shipped head stays a plain matvec.
    """
    if scale is None:
        scale = design_scale(X)
    Xs = X / scale
    G = Xs.T @ Xs
    G[np.diag_indices(G.shape[0])] += lam * len(X)
    beta = np.linalg.solve(G, Xs.T @ y)
    return beta / scale


class MLPHead:
    """One-hidden-layer tanh MLP trained by full-batch Adam, in numpy.

    Deliberately small: the residual has a signal-to-noise ratio of about 0.4,
    so capacity beyond a few hundred parameters buys nothing and costs FLOPs
    at predict time.  Inputs are the *scaled* design columns; the output is a
    scalar residual prediction in the original units.
    """

    def __init__(self, n_in: int, n_hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        sc = 1.0 / math.sqrt(n_in)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * sc
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, 1)) / math.sqrt(n_hidden)
        self.b2 = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[:, 0], h

    def fit(self, X, y, *, epochs: int = 400, lr: float = 3e-2,
            wd: float = 1e-6, batch: int = 65536, seed: int = 0,
            verbose: bool = False):
        rng = np.random.default_rng(seed + 1)
        ps = self.params()
        ms = [np.zeros_like(p) for p in ps]
        vs = [np.zeros_like(p) for p in ps]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        nrow = len(X)
        for ep in range(epochs):
            idx = rng.permutation(nrow)
            for lo in range(0, nrow, batch):
                sel = idx[lo:lo + batch]
                xb, yb = X[sel], y[sel]
                h = np.tanh(xb @ self.W1 + self.b1)
                pred = (h @ self.W2 + self.b2)[:, 0]
                r = pred - yb
                m = len(sel)
                gout = (2.0 / m) * r[:, None]
                gW2 = h.T @ gout
                gb2 = gout.sum(0)
                gh = gout @ self.W2.T * (1.0 - h * h)
                gW1 = xb.T @ gh
                gb1 = gh.sum(0)
                grads = [gW1 + wd * self.W1, gb1, gW2 + wd * self.W2, gb2]
                step += 1
                for p, g, mm, vv in zip(ps, grads, ms, vs):
                    mm *= b1
                    mm += (1 - b1) * g
                    vv *= b2
                    vv += (1 - b2) * g * g
                    p -= lr * (mm / (1 - b1 ** step)) / (
                        np.sqrt(vv / (1 - b2 ** step)) + eps)
            if verbose and (ep + 1) % 50 == 0:
                pr, _ = self.forward(X)
                print(f"    epoch {ep+1:4d}  train mse "
                      f"{float(np.mean((pr - y) ** 2)):.6e}", flush=True)
        return self

    def to_dict(self, prefix: str = "mlp_"):
        return {prefix + "W1": self.W1.astype(np.float32),
                prefix + "b1": self.b1.astype(np.float32),
                prefix + "W2": self.W2.astype(np.float32),
                prefix + "b2": self.b2.astype(np.float32)}
