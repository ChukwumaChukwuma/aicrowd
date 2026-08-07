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

Also NOT here, and measured second: **high-degree Hermite CVs on
network-adapted directions** (``docs/hermite_rank_ceiling.md``,
``scripts/32_adapted_hermite.py``).  The ``p/N`` argument above says nothing
about approximation power, and ``He_d`` of a layer-1 pre-activation is a
degree-``d`` object, so the barrier's ``k <= 2`` instance never bounded
``k >= 3``.  It is still dead, for a third reason: ``h_d(<a,x>)`` is exactly a
unit RANK-ONE tensor of the degree-``d`` chaos, and past degree 2 the
network's chaos content -- 54% of ``Var(y)``, mean Hermite degree 10.5 -- has
no rank-one component.  Measured over 71 dictionaries and 2,494 features
(degree <= 16, mean-field/mean-Jacobian/per-neuron/random direction frames,
all cross products, held-out against an analytic Gram): the family caps at
**43.2%**, ``CV_KMAX = 2`` on the coordinate basis already reaches **39.2%**,
and the argmax of held-out R^2 net of ``p/N`` over the whole surface is
exactly the basis below.

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

#: Direction counts offered to the offline head for the adapted degree-1
#: block.  ``qr`` is column-nested, so ``Q[:, :m]`` for the largest entry
#: supplies every smaller one at no extra cost and the head picks ``m`` on a
#: validation split of GENERATED data.
CVA_GRID: tuple[int, ...] = (8, 16, 24, 32, 48)

#: Direction count the research feature block reports as plain ``cva``.
#: Selected on the validation split of the generated data by
#: ``scripts/28 --mode fit``; the official suite was never consulted for it.
#: NOT shipped -- see :data:`DROPPED`.
CVA_M: int = 48


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
    # ROUND 9: the adapted degree-1 block.  It is a real mechanism -- at unit
    # coefficient over 640 generated MLPs ``-cva24 -cv2`` is 1.371x against
    # ``-cv1 -cv2``'s 1.328x, exactly the 3% the R^2 surface predicted -- and
    # it is still not worth shipping, because the head was ALREADY insuring
    # against the p/N it removes.  Leave-one-group-out on the 18-column
    # design: 1.506x with ``cva``, 1.504x without.  On the official suite it
    # buys 0.32% of raw MSE (3.7039e-6 against 3.7157e-6) and costs 6.2% of
    # C/B, because extracting the directions is 62 flopscope dispatches that
    # nothing else amortises.  ``docs/hermite_rank_ceiling.md`` sec 9.
    "cva", "cva_Phi", "cva_a",
)

#: The round-9 extension.  ``FEATURES_FULL`` is frozen at 28 columns so every
#: ablation table in ``docs/learned_corrector.md`` still reproduces; the
#: adapted degree-1 block is appended here instead.  ``cva`` is the same
#: first-order channel as ``cv1`` and ``cv1mf``, reached a third way: exact
#: split-sample coefficients like ``cv1``, but on ``m`` directions taken from
#: the network instead of all 256, so it pays ``m/N`` of estimation noise
#: rather than ``256/N``.  See ``docs/hermite_rank_ceiling.md`` sec 8.
FEATURES_V2: tuple[str, ...] = FEATURES_FULL + ("cva", "cva_Phi", "cva_a")

#: What the submission actually builds and what the shipped ``beta`` indexes.
FEATURES: tuple[str, ...] = tuple(f for f in FEATURES_V2 if f not in DROPPED)

N_FEATURES = len(FEATURES)
N_FEATURES_FULL = len(FEATURES_FULL)

#: Primitive arrays :func:`sparse_mc_features` returns per output neuron.
PRIMITIVES: tuple[str, ...] = (
    "mu", "cv1", "cv2", "cv3", "cv1mf", "gap", "sk", "ku", "alpha", "Phi",
    "phi", "s", "vh", "sd_mc", "dpilot", "wn", "w4", "gam1", "gam2",
    "cva", *(f"cva{m}" for m in CVA_GRID),
)
#: Per-MLP scalars.
SCALARS: tuple[str, ...] = ("vbar", "arms", "keep_frac")


def meanfield_dirs(weights, gates, m: int):
    """Orthonormal ``(width, m)`` frame spanning the mean-field Jacobian.

    The mean-field input-space Jacobian of ``z^32`` is

        J = W^1 D^1 W^2 D^2 ... D^31 W^32 ,   D^l = diag(Phi(alpha^l))

    and it is very nearly rank one: measured, the top left singular direction
    carries 79-91% of ``||J||_F^2`` and the top 16 carry 99.8%
    (``scripts/32 --mode dirs``).  So a *randomised range finder* recovers its
    column space from a handful of columns, and ``J[:, :m]`` -- the first ``m``
    output neurons' Jacobians, which are generic -- does as well as the exact
    SVD: 23.42% against 23.49% of population span at ``m = 24``, four MLPs.

    Only ``J[:, :m]`` is formed, right to left, so the cost is 31 matmuls of
    ``(width, width) @ (width, m)`` rather than 31 of ``(width, width)^2`` --
    9.8e7 FLOPs at ``m = 24`` instead of 1.0e9.

    A QR then makes the frame orthonormal, which matters twice: the columns of
    ``J`` are nearly parallel (that is the same near-rank-one fact) so their
    Gram has condition ~1e4 and a float32 solve against it would lose most of
    the answer, while an orthonormal frame has population Gram exactly ``I``
    and needs no solve at all.  QR is column-nested, so ``Q[:, :k]`` is the
    frame for ``J[:, :k]`` and one factorisation serves every ``m`` in
    :data:`CVA_GRID`.
    """
    M = weights[-1][:, :m]
    for l in range(len(weights) - 1, 0, -1):
        M = weights[l - 1] @ (gates[l - 1][:, None] * M)
    return np.linalg.qr(M)[0]


def adapted_cv(x, y, Q, split: bool = True):
    """Degree-1 Hermite CV on the orthonormal adapted frame ``Q``.

    ``s = x @ Q`` is exactly standard normal in each coordinate and exactly
    uncorrelated across them, so ``E[s] = 0`` and ``Cov(s) = I`` -- the same
    two facts the layer-1 family rests on, with an analytic Gram that is the
    identity rather than ``rho``.  The correction contracts to two length-N
    matvecs exactly as in :func:`hermite_cv`, and ``split`` uses two halves so
    the ``Cov(g' G^-1 g, y)/N`` self-term of the one-pass form is absent.
    """
    n = x.shape[0]
    s = x @ Q
    if not split:
        d = np.mean(s, axis=0)
        w = s @ d
        return (y.T @ (w - np.mean(w))) / n
    h = n // 2
    s1, s2 = s[:h], s[h:]
    d1, d2 = np.mean(s1, axis=0), np.mean(s2, axis=0)
    wa = s1 @ d2
    wa = wa - np.mean(wa)
    wb = s2 @ d1
    wb = wb - np.mean(wb)
    return 0.5 * ((y[:h].T @ wa) / h + (y[h:].T @ wb) / (n - h))


# ---------------------------------------------------------------------------
# ROUND 12: the exactly-integrable blocks (docs/integrable_cv.md).
#
# ``z^1 = x W^1`` is exactly ``N(0, W^1'W^1)``, so ``E[relu(z^1)]`` is closed
# form and ``Cov(relu(z^1))`` is the arc-cosine kernel -- which makes
# ``E[z^2]`` and ``Cov(z^2)`` exact too.  Those are the LAST exact objects in
# the network (``tests/test_integrable_cv.py`` pins layer 3 as not exact, at
# >20 sigma of 2e6 samples).  The two control variates below live entirely
# inside that region, so their means carry no bias at all.
#
# Measured on 3 local MLPs, held out, net of ``p/N`` at ``N = 27000``
# (``scripts/41_integrable_cv.py --mode quad``):
#
#     shipped t + He_2                      R^2_eff 36.58%   1.577    1.000x
#     relu1_cv alone (256 features)                 37.68%   1.605    1.018x
#     shipped + relu1 + quad2(k=32)                 41.66%   1.714    1.087x
#
# NOT wired into ``FEATURES``: the ``quad2`` frame needs ~60 flopscope
# dispatches (see :func:`kink_frame`), which is the same tax that made ``cva``
# a net loss, and the integration decision belongs with whoever is tuning
# ``N`` against the grader.
# ---------------------------------------------------------------------------
def layer12_moments(weights):
    """``(mh1, Ch1, m2, C2)`` -- the exact layer-1 and layer-2 moments.

    ``x ~ N(0, I)`` exactly, so with ``S = W^1' W^1`` and ``s = sqrt(diag S)``:

        E[relu(z^1_i)]                = s_i / sqrt(2 pi)                exact
        Cov(relu(z^1_i), relu(z^1_j)) = arc-cosine kernel of S_ij/s_i s_j
        E[z^2] = W^2' E[relu(z^1)],  Cov(z^2) = W^2' Cov(relu(z^1)) W^2

    all in closed form.  ``O(n^2)`` elementwise work plus two ``n^3`` matmuls.
    """
    from .relu_moments import relu_cov_exact_centered, relu_var  # noqa: PLC0415
    W1 = np.asarray(weights[0], dtype=np.float64)
    W2 = np.asarray(weights[1], dtype=np.float64)
    S = W1.T @ W1
    s = np.sqrt(np.maximum(np.diag(S), VAR_FLOOR))
    mh1 = s * INV_SQRT_2PI
    Ch1 = relu_cov_exact_centered(S, s)
    np.fill_diagonal(Ch1, relu_var(np.zeros_like(s), s))
    return mh1, Ch1, W2.T @ mh1, W2.T @ Ch1 @ W2


def kink_frame(weights, mz, Cz, k: int):
    """Top-``k`` directions of the degree-2 chaos, in ``z^2`` coordinates.

    ``docs/hermite_rank_ceiling.md`` sec 5.2: the degree-2 chaos of a ReLU
    network is carried entirely by its kink surfaces, neuron ``(l,i)``
    contributing a rank-one term along its own normal ``n_li = grad z^l_i``
    with weight ``E[delta(z^l_i)] dy_j/dz^l_i``.  The weighted second-moment
    matrix of all 8,192 of them,

        Q = sum_{l,i} (phi(alpha_li)/s_li)^2 ||R^l[i,:]||^2 nhat_li nhat_li'

    (``R^l`` the mean-field Jacobian from layer ``l`` to the output), has as its
    top eigenvectors the best *shared* frame for a degree-2 dictionary.
    Measured, it reaches 78-85% of sec 5.2's every-subspace-at-once bound at
    every ``m``, against 69% for the mean-field frame at ``m = 16``.

    ``mz``/``Cz`` are the Gaussian-closure pre-activation moments per layer;
    only ``alpha = m/s`` and ``s`` enter, and only through the *weight*, so
    closure error here costs frame quality and never bias.

    **Cost warning.** The kink weight lives at the deep end -- layers 21-32
    carry 87% of it -- so ``Q`` needs the forward normal recursion AND the
    backward Jacobian sweep over the full depth: ~60 flopscope dispatches,
    ~15 ms, i.e. ~5.5% of the free budget (``hermite_rank_ceiling`` sec 9.4).
    Without the adapted frame the block is worth 1.004x instead of 1.087x, so
    this cost is not optional; it is the mechanism.
    """
    from .relu_moments import phi as _phi  # noqa: PLC0415
    dep = len(weights)
    n = weights[0].shape[1]
    Wf = [np.asarray(w, dtype=np.float64) for w in weights]
    s = [np.sqrt(np.maximum(np.diag(Cz[l]), VAR_FLOOR)) for l in range(dep)]
    al = [mz[l] / s[l] for l in range(dep)]
    g = [norm_cdf(al[l]).astype(np.float64) for l in range(dep)]
    R = [None] * dep
    R[dep - 1] = np.diag(g[dep - 1])
    for l in range(dep - 2, -1, -1):
        R[l] = g[l][:, None] * (Wf[l + 1] @ R[l + 1])
    Q = np.zeros((n, n))
    Ml = None
    for l in range(dep):
        w = (_phi(al[l]) / s[l]) ** 2 * np.sum(R[l] * R[l], axis=1)
        if Ml is not None:
            B = Ml / np.maximum(np.linalg.norm(Ml, axis=0), 1e-30)
            Q += (B * w) @ B.T
        if l + 1 < dep:
            Ml = np.eye(n) if l + 1 == 1 else (Ml * g[l]) @ Wf[l + 1]
    return np.linalg.eigh(Q)[1][:, ::-1][:, :k]


def relu1_cv(h1, y, mh1, Ch1, split: bool = True, jitter: float = 1e-6):
    """Control variate on ``relu(z^1)`` itself -- 256 features, exact mean.

    ``E[relu(z^1_i)] = sigma_i/sqrt(2 pi)`` exactly and the Gram is the
    ANALYTIC arc-cosine matrix, so nothing but ``Cov(feature, y)`` is
    estimated.  This is a *replacement* for the shipped ``t`` + ``He_2`` pair,
    not an addition: measured 1.018x with half the coefficients, and 0.9997x
    when stacked on top of them.  ``relu`` mixes Hermite degrees 1 and 2 in the
    ratio 73.4 : 23.4 of its own variance, which is close to what this target
    wants, so one feature per direction does what two were doing.

    ``Ch1`` inherits the conditioning of ``W^1' W^1`` (~1e8 at this shape), so
    the solve is float64 with a relative diagonal jitter; a float32 solve here
    would lose the answer, exactly as :data:`GRAM_JITTER` notes for ``rho``.
    """
    n = h1.shape[0]
    G = np.array(Ch1, dtype=np.float64, copy=True)
    G[np.diag_indices_from(G)] += jitter * float(np.trace(G)) / len(G)
    gg = np.asarray(h1, dtype=np.float64) - mh1
    return _cv_from_gram(gg, y, G, split)


def quad2_cv(z2, y, m2, C2, A2, split: bool = True):
    """Degree-2 control variate in the layer-1 ACTIVATIONS, exact mean.

    Features ``v_a v_b - (A2' C2 A2)_ab`` with ``v = (z^2 - m^2) A2``.  Because
    ``Cov(z^2)`` is exact (:func:`layer12_moments`), every feature is exactly
    mean zero -- and ``z^2`` is already on hand from the forward pass, so the
    marginal cost is ``k n 2 + k(k+1)/2`` multiplies a sample (1.9e4 FLOPs at
    ``k = 32``, 0.65% of the scored pass).

    The Gram is taken from Wick on the exact ``Cov(z^2)``,
    ``Cov(v_av_b, v_cv_d) = C_ac C_bd + C_ad C_bc``, which assumes ``z^2``
    Gaussian.  It is not, but **that approximation costs efficiency and never
    bias**: the mean of every feature is exact regardless of the Gram, and a
    wrong Gram only mis-weights the correction.  Avoiding it would need an
    ``N p^2`` covariance pass, which at ``p = 528`` is 2.8% of the whole FLOP
    budget -- 40x what the features themselves cost.

    This block is not bounded by the two ceilings that closed everything else:
    ``relu(z_i) relu(z_j)`` is not a polynomial so it escapes
    ``f_1 + f_2 = 45.6%``, and it is not rank-one at degree 2 so it escapes
    ``docs/hermite_rank_ceiling.md`` sec 5.  Measured 15.7% alone at ``k = 32``
    and 1.087x in combination.
    """
    k = A2.shape[1]
    Cv = A2.T @ np.asarray(C2, dtype=np.float64) @ A2
    v = (np.asarray(z2, dtype=np.float64) - m2) @ A2
    iu, ju = np.triu_indices(k)
    gg = v[:, iu] * v[:, ju] - Cv[iu, ju]
    G = (Cv[np.ix_(iu, iu)] * Cv[np.ix_(ju, ju)]
         + Cv[np.ix_(iu, ju)] * Cv[np.ix_(ju, iu)])
    return _cv_from_gram(gg, y, G, split)


def _cv_from_gram(gg, y, G, split: bool):
    """``c_j = Cov(y_j, g' G^-1 gbar)``, split-sample so it is unbiased.

    Shared by :func:`relu1_cv` and :func:`quad2_cv`.  ``G`` is the ANALYTIC
    Gram, so the only estimated quantity is the covariance with the target and
    the ``Cov(g' G^-1 g, y)/N`` self-term of the one-pass form is removed by
    the two-half split exactly as in :func:`hermite_cv`.
    """
    n = len(gg)
    ev, V = np.linalg.eigh(G)
    ev = np.maximum(ev, 1e-12 * max(float(ev.max()), 1e-300))

    def _w(block, d):
        u = V @ ((V.T @ d) / ev)
        w = block @ u
        return w - np.mean(w)

    if not split:
        return (np.asarray(y, dtype=np.float64).T
                @ _w(gg, np.mean(gg, axis=0))) / n
    h = n // 2
    g1, g2 = gg[:h], gg[h:]
    d1, d2 = np.mean(g1, axis=0), np.mean(g2, axis=0)
    y = np.asarray(y, dtype=np.float64)
    return 0.5 * ((y[:h].T @ _w(g1, d2)) / h
                  + (y[h:].T @ _w(g2, d1)) / (n - h))


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
        "cva": f["cva"], "cva_Phi": f["cva"] * Ph, "cva_a": f["cva"] * a,
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

    # ---- adapted degree-1 block, every m in the grid from one QR ---------
    gates = [norm_cdf(a_).astype(np.float32) for a_ in alpha]
    Qmax = meanfield_dirs(weights, gates, max(CVA_GRID))
    cva = {f"cva{m}": adapted_cv(x0, x, Qmax[:, :m], split=split)
           for m in CVA_GRID}

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
        # ``cva`` is the grid member the SHIPPED kernel computes; the rest are
        # carried so the head can pick ``m`` on a validation split without a
        # second pass over the training set.
        "cva": cva[f"cva{CVA_M}"], **cva,
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
