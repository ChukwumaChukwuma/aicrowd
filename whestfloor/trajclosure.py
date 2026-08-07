"""Trajectory-calibrated moment closure for a deep ReLU MLP.

What this is for
================

`docs/integrable_cv.md` reduced the whole deep control-variate programme to one
scalar.  A linear control variate in ``relu(z^L)`` -- 256 features the forward
pass has already computed -- reaches held-out ``R^2`` of 72% at ``L = 8`` and
98.9% at ``L = 32``, against ~39% for the shipped layer-1 basis.  What blocks it
is ``E[g]``: a control variate needs its mean, and for ``L >= 2`` that needs a
moment closure.  A wrong mean does not merely add bias, it multiplicatively
discounts the correction by ``D / (D + b^2)``, and with the Gaussian closure the
entire ``L = 2..32`` ladder is worth 1.03x.

Writing ``r`` for the factor by which a closure's mean error beats the Gaussian
closure's, break-even is ``r = 4.2-4.6`` at every depth from 6 to 32.  This
module is the attempt to reach it.

The method, and its provenance
==============================

The construction is **jamesrahenry's trajectory-calibrated moment chain**
(AIcrowd discourse topic 18097, "Stabilizing cumulant propagation at depth 32",
graded submission #314695; MIT replication repository
``github.com/jamesrahenry/arc-whitebox-replication``).  Its diagnosis is the
part worth restating:

    A deep moment-propagation chain is an error-compensating dynamical system.
    Per-layer errors are anticorrelated and partially cancel, so the chain
    attenuates zero-mean noise but AMPLIFIES coherent bias (measured ~16:1).
    Consequently step-exact Edgeworth corrections computed from *true* cumulants
    make the chained estimator worse, and corrections work only when fitted on
    the chain's own rolled-forward trajectories -- DAgger [Ross et al. 2011] /
    scheduled sampling, transplanted from imitation learning to a deterministic
    analytic recursion.

Their fitting targets came from keenanpepper's public per-layer moment atlas
(``keenanpepper/arc-whestbench-higher-moments-2026``) and their ground-truth
joint-cumulant tests from ``keenanpepper/whest-k3-tensors-2026``.  Both are
credited in ``docs/traj_closure.md``.

What is different here, and why it has to be
============================================

Their chain is **probe-fed**: five of its eight mean features, three of its six
variance features and both of its off-diagonal features are built from an
``N = 4096`` plain-MC probe of the target network's own per-layer cumulants
(``kappa_3``, ``kappa_4`` and the pair field ``E[zc_i^2 zc_j]``), costing ~6% of
the FLOP budget.

**A probe-fed mean is worth nothing as a control-variate mean.**  A control
variate on ``h^L`` with a claimed mean is exactly a shrinkage between the
analytic mean and the sample mean of that layer's activations
(``docs/integrable_cv.md`` sec 3.3), so it pays only where the analytic error
beats the ``N``-sample noise of the pass that is *already running*.  If the
claimed mean is itself estimated from ``N_p`` fresh samples with variance
``V/N_p``, the control variate's variance picks that up in full and the whole
device is a loss unless ``N_p >> N`` -- at which point the probe, not the
closure, is the estimator.  So the deployable arm here uses **probe-free
features only**, and every cumulant it consumes is analytic: the star / tree
diagram source of ``docs/cumulant_expansion.md`` sec 5, optionally closed with
the diagonal transport of that document's sec 9.6.

The probe-fed variants are still implemented -- as *ceilings*.  ``src="oracle"``
feeds exact Monte-Carlo cumulants, which is jamesrahenry's "perfect-cumulant
control" and upper-bounds every probe.

Layout
======

``closure_step``    one Gaussian-closure layer: post-ReLU mean/covariance from
                    pre-activation mean/covariance.  Layer 1 uses the exact
                    arc-cosine kernel, so the first two layers are exact.
``analytic_cum``    per-neuron ``kappa_3``/``kappa_4`` of a pre-activation from
                    the previous layer's Hermite coefficients: star diagram,
                    tree catalogue, or either plus diagonal transport.
``mean_features``   the design matrix of the per-layer mean correction.
``var_features``    ... of the variance correction.
``offdiag_features``... of the covariance correction.
``chain``           the driver: rolls the state forward, applying whatever
                    per-layer coefficients it is given, and hands back
                    everything a DAgger fit needs at each layer.
"""

from __future__ import annotations

import math

import numpy as np

from .relu_moments import (
    Phi,
    phi,
    relu_cov_exact_centered,
    relu_cov_mehler,
    relu_hermite_coeffs,
    relu_mean,
    relu_var,
)

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
VAR_FLOOR = 1e-30

#: Mehler truncation.  ``docs/cumulant_expansion.md`` sec 10.1: the two-block
#: sums are edge-kernel folded, so raising this costs ``O(K n^2)`` and no
#: matmuls.
KMAX = 8

# ---------------------------------------------------------------------------
# Feature names.  A "spec" is an ordered tuple of these.
# ---------------------------------------------------------------------------
#: Mean-correction features that need no cumulant at all.
MEAN_FREE = ("sphi", "sphi_a", "sphi_a2", "sphi_a3", "sPhi", "mu0")
#: Mean-correction features built from a per-neuron third/fourth cumulant.
MEAN_CUM = ("k3E", "k4E", "k3E_a", "k4E_a", "k3n")
#: jamesrahenry's eight, in his order (his 1-3 are k3E, k4E, sphi).
MEAN_JRH = ("k3E", "k4E", "sphi", "sphi_a", "sphi_a2", "k3E_a", "k4E_a", "k3n")

VAR_FREE = ("vphi", "vphi_a", "vphi_a2")
VAR_CUM = ("k3V", "k3n_v", "k4n_v")
VAR_JRH = ("k3V", "vphi", "vphi_a", "vphi_a2", "k3n_v", "k4n_v")

OFF_JRH = ("T1", "T1rho")


# ---------------------------------------------------------------------------
# One closure step
# ---------------------------------------------------------------------------
def closure_step(m, C, kmax=KMAX, exact_acos=False):
    """Gaussian-closure post-ReLU moments of ``z ~ (m, C)``.

    Returns ``(mu0, Ch0, sig, alpha, pa, Pa, rho, acoef)``.  ``exact_acos``
    replaces the Mehler series by the closed-form arc-cosine kernel, which is
    exact when ``m == 0`` (layer 1, where ``z^1 = x W^1`` is exactly Gaussian
    with mean zero).
    """
    C = 0.5 * (C + C.T)
    var = np.maximum(np.diag(C), VAR_FLOOR)
    sig = np.sqrt(var)
    alpha = m / sig
    pa = phi(alpha)
    Pa = Phi(alpha)
    mu0 = m * Pa + sig * pa
    if exact_acos:
        Ch0 = relu_cov_exact_centered(C, sig)
    else:
        Ch0 = relu_cov_mehler(C, m, sig, kmax=kmax)
    np.fill_diagonal(Ch0, relu_var(m, sig))
    rho = np.clip(C / np.outer(sig, sig), -0.9999, 0.9999)
    acoef = relu_hermite_coeffs(m, sig, max(kmax, 4))
    return mu0, Ch0, sig, alpha, pa, Pa, rho, acoef


# ---------------------------------------------------------------------------
# Analytic per-neuron cumulants of the NEXT pre-activation
# ---------------------------------------------------------------------------
def kappa_x(acoef, mu0, sig, alpha, pa, Pa):
    """Marginal ``kappa_2..4`` of ``x = relu(z)`` under the Gaussian closure.

    Truncated-normal moments ``I_p = E[(t+alpha)^p 1{t > -alpha}]`` give the
    raw moments of the rectifier in closed form
    (``docs/cumulant_expansion.md`` Corollary A').
    """
    I0 = Pa
    I1 = alpha * Pa + pa
    I2 = alpha * I1 + I0
    I3 = alpha * I2 + 2.0 * I1
    I4 = alpha * I3 + 3.0 * I2
    s2 = sig * sig
    m1 = sig * I1
    m2 = s2 * I2
    m3 = s2 * sig * I3
    m4 = s2 * s2 * I4
    k2 = m2 - m1 * m1
    k3 = m3 - 3.0 * m2 * m1 + 2.0 * m1 ** 3
    k4 = m4 - 4.0 * m3 * m1 - 3.0 * m2 * m2 + 12.0 * m2 * m1 * m1 - 6.0 * m1 ** 4
    return k2, k3, k4


def kappa3_star_np(W, acoef, rho, umax=1):
    """Star-diagram ``kappa_3`` of ``z'_j = sum_i W_ij x_i``.

    The ``(1,1,1)`` slot partition of ``docs/cumulant_expansion.md`` sec 5.2
    with no injectivity correction, contracted against ``R`` -- the incomplete
    form, kept because it is what ``whestfloor/kernels.cov_prop_edgeworth``
    ships and therefore what the ``kappa_3`` arm's published ``r = 1.45``
    refers to.
    """
    fact = [1.0]
    for i in range(1, 2 * umax + 2):
        fact.append(fact[-1] * i)
    G = [None] + [acoef[k][:, None] * W for k in range(1, 2 * umax + 1)]
    Rp = [None, rho]
    for u in range(2, umax + 1):
        Rp.append(Rp[u - 1] * rho)
    RG = [None] + [Rp[u] @ G[u] for u in range(1, umax + 1)]
    acc = 0.0
    for u in range(1, umax + 1):
        for v in range(1, umax + 1):
            acc = acc + np.sum(G[u + v] * RG[u] * RG[v], axis=0) / (
                fact[u] * fact[v])
    return 3.0 * acc


def analytic_cum(W, m, sig, rho, acoef, k3x, k4x,
                 src="star", umax=1, K2=8, T3=3, T4=3):
    """``(kappa_3, kappa_4)`` of the NEXT pre-activation, analytically.

    ``src`` selects the diagram catalogue:

    ``"coinc"``  the all-coincident ``(3)``/``(4)`` diagrams alone,
                 ``(W^oa)' kappa_a(x)`` -- the transport-only limit, 0 matmuls.
    ``"star"``   ``whestfloor.cumulants.kappa3_star``, the incomplete
                 ``(1,1,1)`` form the shipped ``cov_prop_edgeworth`` arm uses.
                 This is what the published ``r = 1.45`` refers to.
    ``"tree"``   ``whestfloor.cumulants.kappa3_tree`` / ``kappa4_tree`` -- the
                 complete tree catalogue with the injectivity corrections and
                 both Ursell terms, verified in ``scripts/13`` and ``scripts/16``
                 against Monte Carlo to within one standard error.
    """
    from .cumulants import coeffs, epow, kappa3_tree, kappa4_tree  # noqa: PLC0415
    W2 = W * W
    k4 = (W2 * W2).T @ k4x
    if src == "coinc":
        return (W2 * W).T @ k3x, k4
    if src == "star":
        return kappa3_star_np(W, acoef, rho, umax=umax), k4
    if src in ("tree", "tree3"):
        K = max(K2, T3, T4) + 1
        ctx = coeffs(m, sig, K) + (epow(rho, K),)
        k3 = kappa3_tree(W, m, sig, rho, K2=K2, T3=T3, ctx=ctx)
        if src == "tree3":
            return k3, k4
        return k3, kappa4_tree(W, m, sig, rho, K2=K2, T4=T4, ctx=ctx)
    raise ValueError(src)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _mean_feature(name, sig, alpha, pa, Pa, mu0, k3, k4):
    var = sig * sig
    if name == "sphi":
        return sig * pa
    if name == "sphi_a":
        return sig * pa * alpha
    if name == "sphi_a2":
        return sig * pa * alpha * alpha
    if name == "sphi_a3":
        return sig * pa * alpha ** 3
    if name == "sPhi":
        return sig * Pa
    if name == "mu0":
        return mu0
    if name == "one":
        return np.ones_like(sig)
    if name == "k3E":
        return -(k3 / 6.0) * (alpha * pa / var)
    if name == "k4E":
        return (k4 / 24.0) * (pa * (alpha * alpha - 1.0) / (sig * var))
    if name == "k3E_a":
        return -(k3 / 6.0) * (alpha * pa / var) * alpha
    if name == "k4E_a":
        return (k4 / 24.0) * (pa * (alpha * alpha - 1.0) / (sig * var)) * alpha
    if name == "k3n":
        return (k3 / var) * pa
    raise ValueError(name)


def _var_feature(name, sig, alpha, pa, Pa, k3, k4):
    var = sig * sig
    if name == "vphi":
        return var * pa
    if name == "vphi_a":
        return var * alpha * pa
    if name == "vphi_a2":
        return var * alpha * alpha * pa
    if name == "vone":
        return var
    if name == "k3V":
        return k3 * pa / (3.0 * sig) - k4 * alpha * pa / (12.0 * var)
    if name == "k3n_v":
        return k3 * pa / sig
    if name == "k4n_v":
        return k4 * pa / var
    raise ValueError(name)


def mean_features(spec, sig, alpha, pa, Pa, mu0, k3, k4):
    return np.stack([_mean_feature(nm, sig, alpha, pa, Pa, mu0, k3, k4)
                     for nm in spec], axis=1)


def var_features(spec, sig, alpha, pa, Pa, k3, k4):
    return np.stack([_var_feature(nm, sig, alpha, pa, Pa, k3, k4)
                     for nm in spec], axis=1)


def offdiag_features(spec, m, sig, alpha, pa, rho, kiij):
    """jamesrahenry's ``T1``: the pair field gated by the conditional CDF.

    ``T1_ik = sym( kappa_iij * (phi(alpha_i)/sigma_i) Phi(m_cond/s_cond) )`` is
    the first-order Edgeworth response of ``Cov(relu)`` to the joint third
    cumulant of the pre-activation, evaluated through the conditional-Gaussian
    structure of the closure.  It is the one feature his ablation found
    irreplaceable.
    """
    if kiij is None:
        return []
    s_cond = sig[None, :] * np.sqrt(np.clip(1.0 - rho * rho, 1e-12, None))
    m_cond0 = m[None, :] - rho * sig[None, :] * alpha[:, None]
    G = (pa[:, None] / sig[:, None]) * Phi(m_cond0 / s_cond)
    KG = kiij * G
    T1 = 0.5 * (KG + KG.T)
    out = []
    for nm in spec:
        if nm == "T1":
            out.append(T1)
        elif nm == "T1rho":
            out.append(T1 * rho)
        else:
            raise ValueError(nm)
    return out


def analytic_kiij(W, k3x):
    """``kappa_iij(z') = (W^o2)' diag(kappa_3(x)) W`` -- the coincident-block
    order of the mixed third cumulant (``docs/cumulant_expansion.md`` sec 7).
    One matmul, and it is the probe-free stand-in for the pair field."""
    W2 = W * W
    return W2.T @ (k3x[:, None] * W)


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
def chain(W, *, mspec=(), vspec=(), ospec=(), coefs=None, src="star",
          ext_cum=None, kmax=KMAX, umax=1, K2=8, T3=3, T4=3, want_off=False,
          exact_layer1=True, transport=None):
    """Roll the closure forward, yielding one record per layer.

    ``coefs[l] = (Am, Av, Ao)`` are the per-layer correction coefficients; pass
    ``None`` for the uncorrected chain.  ``ext_cum[l] = (k3, k4, kiij)``
    supplies externally measured cumulants of ``z^{l+1}`` (probe or oracle) in
    place of the analytic ones.

    ``transport`` closes the cumulant recursion of
    ``docs/cumulant_expansion.md`` sec 9.6 at diagonal order: with
    ``transport = (g3, g4)``,

        kappa_3(z^{l+1}) = source + g3 * (diag(Phi) W)^{o3}' kappa_3(z^l)

    which is one elementwise-cubed matvec (``2 n^2`` FLOPs) and is the term that
    document identifies as the reason the diagram source alone produces only
    ~15% of the observed skewness at depth.

    Yields, for ``l = 0 .. depth-1``, a dict with the pre-activation moments,
    the raw closure output, the design matrices and the corrected state.  The
    corrected post-ReLU mean is ``rec["mu"]``, i.e. ``E[relu(z^{l+1})]``.
    """
    n = W[0].shape[1]
    m = np.zeros(n)
    C = W[0].astype(np.float64).T @ W[0].astype(np.float64)
    depth = len(W)
    _prev_cum = (np.zeros(n), np.zeros(n))
    _prev_kiij = np.zeros((n, n)) if want_off else None
    for l in range(depth):
        rec_m = m
        mu0, Ch0, sig, alpha, pa, Pa, rho, acoef = closure_step(
            m, C, kmax=kmax, exact_acos=(l == 0 and exact_layer1))
        _, k3x, k4x = kappa_x(acoef, mu0, sig, alpha, pa, Pa)
        # cumulants OF THIS LAYER's pre-activation z^{l+1}: for l == 0 the
        # input is exactly Gaussian, so they are exactly zero.
        if ext_cum is not None:
            k3, k4, kiij = ext_cum[l]
        else:
            k3, k4 = _prev_cum
            kiij = _prev_kiij
        Fm = (mean_features(mspec, sig, alpha, pa, Pa, mu0, k3, k4)
              if mspec else np.zeros((n, 0)))
        Fv = (var_features(vspec, sig, alpha, pa, Pa, k3, k4)
              if vspec else np.zeros((n, 0)))
        Fo = (offdiag_features(ospec, m, sig, alpha, pa, rho, kiij)
              if (ospec and kiij is not None) else [])
        mu = mu0.copy()
        Ch = Ch0
        if coefs is not None:
            Am, Av, Ao = coefs[l]
            if len(Am):
                mu = mu + Fm @ Am
            if len(Av) or len(Ao):
                Ch = Ch0.copy()
                for c, f in zip(Ao, Fo):
                    Ch = Ch + c * f
                dg = np.diag(Ch0).copy()
                if len(Av):
                    dg = dg + Fv @ Av
                np.fill_diagonal(Ch, np.maximum(dg, VAR_FLOOR))
        yield {
            "l": l, "m": m, "C": C, "sig": sig, "alpha": alpha, "pa": pa,
            "Pa": Pa, "rho": rho, "mu0": mu0, "Ch0": Ch0, "Fm": Fm, "Fv": Fv,
            "Fo": Fo, "mu": mu, "Ch": Ch, "acoef": acoef, "k3x": k3x,
            "k3": k3, "k4": k4,
        }
        if l + 1 < depth:
            Wn = W[l + 1].astype(np.float64)
            m = Wn.T @ mu
            C = Wn.T @ Ch @ Wn
            if ext_cum is None:
                s3, s4 = analytic_cum(Wn, rec_m, sig, rho, acoef, k3x, k4x,
                                      src=src, umax=umax, K2=K2, T3=T3, T4=T4)
                if transport is not None:
                    g3, g4 = transport
                    Wt = Pa[:, None] * Wn
                    s3 = s3 + g3 * ((Wt ** 3).T @ k3)
                    s4 = s4 + g4 * ((Wt ** 4).T @ k4)
                _prev_cum = (s3, s4)
                _prev_kiij = analytic_kiij(Wn, k3x) if want_off else None


def run(W, **kw):
    """Convenience: the corrected ``E[relu(z^l)]`` for every layer, ``(D, n)``."""
    return np.stack([rec["mu"] for rec in chain(W, **kw)], axis=0)


class ChainDriver:
    """The same recursion, driven one layer at a time.

    A DAgger fit needs the layer-``l`` design matrices *before* layer ``l``'s
    coefficients exist, so the chain has to be suspended between the closure
    step and the correction.  ``step()`` advances to the next layer and returns
    its record; ``apply(Am, Av, Ao)`` installs that layer's coefficients and
    commits the state.  Rolling every training net forward this way is what
    makes the fit's input distribution the deployment distribution.
    """

    def __init__(self, W, *, mspec=(), vspec=(), ospec=(), src="star",
                 ext_cum=None, kmax=KMAX, umax=1, K2=8, T3=3, T4=3,
                 want_off=False, exact_layer1=True, transport=None):
        self.W = W
        self.n = W[0].shape[1]
        self.depth = len(W)
        self.mspec, self.vspec, self.ospec = mspec, vspec, ospec
        self.src, self.ext_cum = src, ext_cum
        self.kmax, self.umax, self.K2, self.T3, self.T4 = kmax, umax, K2, T3, T4
        self.want_off = want_off or bool(ospec)
        self.exact_layer1 = exact_layer1
        self.transport = transport
        self.l = 0
        self.m = np.zeros(self.n)
        self.C = W[0].astype(np.float64).T @ W[0].astype(np.float64)
        self._cum = (np.zeros(self.n), np.zeros(self.n))
        self._kiij = np.zeros((self.n, self.n)) if self.want_off else None
        self._rec = None
        self.last_mu = None

    def step(self):
        l = self.l
        mu0, Ch0, sig, alpha, pa, Pa, rho, acoef = closure_step(
            self.m, self.C, kmax=self.kmax,
            exact_acos=(l == 0 and self.exact_layer1))
        _, k3x, k4x = kappa_x(acoef, mu0, sig, alpha, pa, Pa)
        if self.ext_cum is not None:
            k3, k4, kiij = self.ext_cum[l]
        else:
            k3, k4 = self._cum
            kiij = self._kiij
        Fm = (mean_features(self.mspec, sig, alpha, pa, Pa, mu0, k3, k4)
              if self.mspec else np.zeros((self.n, 0)))
        Fv = (var_features(self.vspec, sig, alpha, pa, Pa, k3, k4)
              if self.vspec else np.zeros((self.n, 0)))
        Fo = (offdiag_features(self.ospec, self.m, sig, alpha, pa, rho, kiij)
              if (self.ospec and kiij is not None) else [])
        self._rec = {"l": l, "m": self.m, "sig": sig, "alpha": alpha, "pa": pa,
                     "Pa": Pa, "rho": rho, "mu0": mu0, "Ch0": Ch0, "Fm": Fm,
                     "Fv": Fv, "Fo": Fo, "acoef": acoef, "k3x": k3x,
                     "k4x": k4x, "k3": k3, "k4": k4}
        return self._rec

    def apply(self, Am, Av, Ao):
        r = self._rec
        mu = r["mu0"] + (r["Fm"] @ Am if len(Am) else 0.0)
        Ch = r["Ch0"]
        if len(Av) or len(Ao):
            Ch = r["Ch0"].copy()
            for c, f in zip(Ao, r["Fo"]):
                Ch = Ch + c * f
            dg = np.diag(r["Ch0"]).copy()
            if len(Av):
                dg = dg + r["Fv"] @ Av
            np.fill_diagonal(Ch, np.maximum(dg, VAR_FLOOR))
        self.last_mu = mu
        l = self.l
        if l + 1 < self.depth:
            Wn = self.W[l + 1].astype(np.float64)
            self.m = Wn.T @ mu
            self.C = Wn.T @ Ch @ Wn
            if self.ext_cum is None:
                s3, s4 = analytic_cum(Wn, r["m"], r["sig"], r["rho"],
                                      r["acoef"], r["k3x"], r["k4x"],
                                      src=self.src, umax=self.umax,
                                      K2=self.K2, T3=self.T3, T4=self.T4)
                if self.transport is not None:
                    g3, g4 = self.transport
                    Wt = r["Pa"][:, None] * Wn
                    s3 = s3 + g3 * ((Wt ** 3).T @ r["k3"])
                    s4 = s4 + g4 * ((Wt ** 4).T @ r["k4"])
                self._cum = (s3, s4)
                self._kiij = (analytic_kiij(Wn, r["k3x"])
                              if self.want_off else None)
        self.l += 1
        return mu
