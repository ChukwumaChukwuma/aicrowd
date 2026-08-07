"""Bit-sliced, stochastically-quantised forward pass.

Forum 18125: @dipam, for the AIcrowd team, confirmed that bit-packing is
**a legitimate optimisation** and that the billing will not be changed for it,
"because we cannot distinguish it -- a packed word and an ordinary integer
array are byte-for-byte identical with the same dtype".  So a ``uint32``
holding 32 boolean lanes costs one billed FLOP per ``bitwise_and``, and a
``b_a x b_w``-bit product over ``n`` inputs can in principle undercut the
``2n`` of a float32 contraction.

This module has two halves that must never be confused:

``np_*``
    A raw-NumPy *research* simulator.  It never runs inside a BudgetContext
    and is only used to sweep precision schedules cheaply.  It reproduces the
    ARITHMETIC of the packed kernel exactly (bit-slicing an integer product is
    an identity, not an approximation), so its variance numbers transfer.
``bitsliced_*`` / ``*_kernel``
    The real flopscope path.  Everything it does is billed.

The three facts the design is built on
--------------------------------------
1. **Billing.**  Measured in a real ``BudgetContext`` (``scripts/44``,
   ``--mode probe``): ``bitwise_and`` and ``bitwise_count`` each cost 1 FLOP
   per uint32 *element* (32 lanes), and an axis reduction of length ``w``
   costs ``w-1`` per output at the accumulator's dtype rate -- so the
   accumulator MUST be declared int32/uint32, since the uint64 default is
   billed at rate 2.0.  A packed dot over ``n`` lanes therefore costs
   ``3n/32`` and not ``2n/32``: **the popcount reduction is a third of the
   bill and the ceiling is 21.3x, not 32x.**
2. **Activation error is variance; weight error is BIAS.**  Stochastic
   rounding of the activations is fresh per sample, so it is zero-mean and
   divides by ``N`` like any other Monte-Carlo variance -- that is the trade
   the score prices.  A quantised WEIGHT is fixed for the whole run: its error
   does not divide by ``N`` at all.  ``ROUND_W_MEANFIX`` repairs the
   first-order part of that by folding ``hbar @ (W - W_hat)`` -- one
   ``(1,K) @ (K,K)`` product per layer per batch, i.e. ``2K^2/N`` a sample --
   into the layer bias, which converts the leading weight-quantisation bias
   into a per-sample fluctuation.
3. **Range, not precision, sets the noise.**  Injected variance is
   ``range^2 / (6 (2^b - 1)^2)`` per element (``E[u(1-u)] = 1/6`` for
   stochastic rounding).  The range is per-neuron and comes from the pilot's
   own ``(m, s)`` of the pre-activation, clipped at zero on the left because
   ``relu >= 0``.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Ranges
# --------------------------------------------------------------------------


def np_ranges(m_z, s_z, kappa):
    """Per-neuron quantisation window ``[lo, hi]`` for ``h = relu(z)``.

    ``z ~ (m, s)`` per neuron, so ``h`` lives in ``[max(0, m - k s),
    max(0, m + k s)]`` up to a ``k``-sigma tail.  Taking the window on the
    PRE-activation scale is what makes it tight for the deep layers: by layer
    32 ``rms|alpha| = 3.44``, the neuron is almost always on, and
    ``[m - k s, m + k s]`` is ``2k s`` wide where the naive ``[0, m + k s]``
    would be ``(alpha + k) s`` wide.  Clipping at zero on the left costs
    nothing because ``relu`` never goes there.
    """
    lo = np.maximum(0.0, m_z - kappa * s_z)
    hi = np.maximum(0.0, m_z + kappa * s_z)
    return lo, hi


# --------------------------------------------------------------------------
# Quantisers (research simulator)
# --------------------------------------------------------------------------


def np_quant_act(h, lo, step, levels, rng, stochastic=True, anti=False):
    """Stochastic rounding of ``h`` onto ``lo + step * {0..levels}``.

    Returns the integer codes as float32 (exact: the codes are small integers
    and every downstream product stays far inside float32's 24-bit mantissa,
    which ``scripts/44 --mode probe`` asserts against an int64 matmul).

    ``anti=True`` draws the uniform for the second half of the batch as
    ``1 - u`` of the first.  The rounding errors of an antithetic pair have
    correlation ``-1/2`` (``E[max(0, f1+f2-1)] - E[f1 f2] = 1/6 - 1/4``
    against a per-draw variance of ``1/6``), so the pair mean carries HALF the
    quantisation variance of two independent draws, for zero extra FLOPs --
    it also halves the number of uniforms that have to be drawn.
    """
    t = (h - lo) / step
    if not stochastic:
        q = np.rint(t)
    else:
        n = t.shape[0]
        if anti:
            half = (n + 1) // 2
            u0 = rng.random(size=(half,) + t.shape[1:], dtype=np.float32)
            u = np.concatenate([u0, 1.0 - u0[: n - half]], axis=0)
        else:
            u = rng.random(size=t.shape, dtype=np.float32)
        q = np.floor(t + u)
    return np.clip(q, 0.0, float(levels)).astype(np.float32)


def np_quant_weight(w, bw, rng, stochastic=True):
    """Per-output-column symmetric quantiser, max-scaled so nothing clips.

    ``levels = 2^(bw-1) - 1`` each side.  Max-scaling (rather than a sigma
    multiple) removes clipping entirely, which matters far more here than the
    range it wastes: a clipped weight is a permanent, sample-independent
    perturbation of the network and therefore pure bias.
    """
    lv = float(2 ** (bw - 1) - 1)
    amax = np.max(np.abs(w), axis=0)
    step = np.where(amax > 0, amax / lv, 1.0).astype(np.float32)
    t = w / step
    if stochastic:
        q = np.floor(t + rng.random(size=t.shape, dtype=np.float32))
    else:
        q = np.rint(t)
    q = np.clip(q, -lv - 1.0, lv)
    return q.astype(np.float32), step


# --------------------------------------------------------------------------
# The research forward pass
# --------------------------------------------------------------------------


def np_group_steps(lo, hi, ba, n_groups):
    """Per-neuron step, constrained to ``base * 2^-g`` with ``g < n_groups``.

    **This constraint is what makes per-neuron scaling implementable at all.**
    A bit-sliced product returns the integer ``sum_i q_i qw_ij``; a per-input
    scale ``step_i`` cannot be pulled out of that sum.  Restricting ``step_i``
    to octaves of a common base lets the neurons be partitioned into
    ``n_groups`` groups, each contracted separately and combined with one
    power-of-two-weighted add per group -- ``2 FLOPs`` an output per group,
    against ``b_a b_w (3w-1)`` for the core.  ``n_groups = 1`` is a single
    per-layer scale; large ``n_groups`` approaches free per-neuron scaling.

    Steps round UP to the octave (``floor`` of the log-ratio), so a neuron's
    window is never narrower than ``[lo, hi]`` and the octave grouping adds no
    clipping of its own.
    """
    rng_i = np.maximum(hi - lo, 1e-30)
    base = float(np.max(rng_i))
    lv = float(2 ** ba - 1)
    g = np.floor(np.log2(base / rng_i))
    g = np.clip(g, 0.0, float(n_groups - 1))
    return (base * np.exp2(-g) / lv).astype(np.float32), g.astype(np.int32)


_SQRT_2PI = 2.5066282746310002


def _phi(a):
    return np.exp(-0.5 * a * a) / _SQRT_2PI


def _Phi(a):
    return 0.5 * (1.0 + _erf(a / 1.4142135623730951))


def _erf(x):
    """Abramowitz-Stegun 7.1.26 -- 1.5e-7 absolute, ample for a noise model."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def np_noise_plan(weights, stats, cfg, steps_by_layer, wresid):
    """Analytic propagation of the quantisation noise, and the mean it costs.

    **This is the mechanism that decides the whole lane.**  Stochastic rounding
    is unbiased through a LINEAR map.  A ReLU network is not one: ``relu`` is
    convex, so an injected zero-mean perturbation ``eps`` of variance ``v`` on
    a pre-activation ``z ~ N(m, s^2)`` raises the mean of the activation by

        E[relu(z + eps)] - E[relu(z)]  =  (1/2) v phi(m/s) / s  +  O(v^2)

    -- a FIRST-ORDER bias in the injected variance, at every one of 32 layers,
    each amplified by everything downstream.  Measured, it is 1.6e-2 rms at
    ``b_a = 4``, and ``0.1 b^2`` is then 100x the entire shipped score.  It is
    not a defect of the rounding; it is the price of injecting variance into a
    convex map, and it is what makes "quantisation is free because it is
    unbiased" wrong for this problem.

    It is also entirely computable in advance.  ``v`` obeys a linear recursion,

        v_pre^{l}_j  =  sum_i [ Phi_i v_pre^{l-1}_i + fresh_i ] W_ij^2  +  wq_j

    (a perturbation survives a relu only through an open gate, hence ``Phi``;
    ``fresh_i = Phi_i step_i^2 / 6`` is the layer's own rounding, with the
    ``Phi`` because ``relu`` outputs an exact zero that the grid represents
    exactly and so does not round), and every input to it -- the steps, the
    pilot's ``(m, s)``, ``W^2`` -- is known per MLP.  The correction
    ``dh^l = (1/2) v_pre^l phi(alpha^l) / s^l`` is therefore a per-MLP
    ``O(depth * K^2)`` computation, ~230 FLOPs a sample at N=27000, and it
    folds into the next layer's bias vector for free.
    """
    m_z, s_z = stats
    depth = len(weights)
    v_prev = None
    v_pre, dh = [], []
    for l in range(depth):
        w = weights[l]
        step = steps_by_layer[l]
        if step is None:                                   # exact layer
            fresh = np.zeros(w.shape[0], dtype=np.float64)
        else:
            gate = (np.ones(w.shape[0]) if l == 0
                    else _Phi(m_z[l - 1] / s_z[l - 1]).astype(np.float64))
            fresh = gate * (step.astype(np.float64) ** 2) / 6.0
        tv = fresh if v_prev is None else v_prev + fresh
        v = (tv[None, :] @ (w.astype(np.float64) ** 2))[0]
        e = wresid[l]
        if e is not None:
            # the weight residual's own per-sample fluctuation: the batch-mean
            # fix removes its mean, what is left is Var(h_i) E_ij^2
            varh = _relu_var(m_z[l - 1], s_z[l - 1]) if l else np.ones(w.shape[0])
            v = v + (varh[None, :] @ (e.astype(np.float64) ** 2))[0]
        a = (m_z[l] / s_z[l]).astype(np.float64)
        v_pre.append(v)
        dh.append(0.5 * v * _phi(a) / s_z[l].astype(np.float64))
        v_prev = _Phi(a) * v
    return v_pre, dh


def _relu_var(m, s):
    a = (m / s).astype(np.float64)
    P, p = _Phi(a), _phi(a)
    mu = m * P + s * p
    e2 = (m.astype(np.float64) ** 2 + s.astype(np.float64) ** 2) * P + m * s * p
    return np.maximum(e2 - mu ** 2, 0.0)


def np_plan(weights, stats, cfg, seed):
    """Everything that is fixed for one MLP: windows, steps, quantised weights.

    Built ONCE per MLP with its own generator, so replicating the experiment
    re-draws only the per-sample activation rounding.  That separation is not
    cosmetic: the weight rounding is fixed for the whole run, so it is a BIAS
    and must not be allowed to average away across replicates.
    """
    m_z, s_z = stats
    rng = np.random.default_rng(seed)
    los, steps, lvls, whats, resid = [], [], [], [], []
    for l, w in enumerate(weights):
        ba, bw = cfg.ba_at(l), cfg.bw_at(l)
        if ba is None or bw is None:
            los.append(None); steps.append(None); lvls.append(None)
            whats.append(w); resid.append(None)
            continue
        if l == 0:                              # input is N(0,1), not a relu
            k0 = cfg.kappa0 or cfg.kappa
            lo = np.full(w.shape[0], -k0, dtype=np.float32)
            hi = np.full(w.shape[0], k0, dtype=np.float32)
        else:
            lo, hi = np_ranges(m_z[l - 1], s_z[l - 1], cfg.kappa)
            lo, hi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        step, _ = np_group_steps(lo, hi, ba, cfg.groups)
        los.append(lo.astype(np.float32))
        steps.append(step)
        lvls.append(float(2 ** ba - 1))
        if bw >= 24:
            whats.append(w)
            resid.append(None)
        else:
            qw, ws = np_quant_weight(w, bw, rng, stochastic=cfg.stoch_w)
            what = qw * ws[None, :]
            whats.append(what)
            resid.append(w - what)
    dh = None
    if cfg.relufix:
        _, dh = np_noise_plan(weights, stats, cfg, steps, resid)
    return {"lo": los, "step": steps, "lv": lvls, "what": whats,
            "resid": resid, "dh": dh, "w": weights,
            "m": m_z.astype(np.float64), "s": s_z.astype(np.float64)}


def np_calibrate(x_cal, plan, cfg, seed, reps=1):
    """Per-layer mean-shift correction, measured instead of modelled.

    The analytic first-order correction ``(1/2) v phi(alpha)/s`` is wrong here
    by up to 3x (``scripts/44``, the propagation diagnostic) because the
    injected perturbation is not small: at ``b_a = 4`` its sd is **half** the
    pre-activation sd at every layer, so nothing linearised in it converges.
    This measures the shift instead, by running the exact and the quantised
    pass on the SAME calibration inputs and taking the paired difference layer
    by layer, correcting as it goes so each layer's estimate is conditional on
    the ones before it being already fixed.

    It is an ORACLE in the sense that matters for a bound: give it as many
    calibration samples as you like and it is the best any bias correction can
    do, because it is a direct unbiased measurement of the thing to subtract.
    A shippable version pays ``n_cal / N`` of a dense pass for it.
    """
    he = np.asarray(x_cal, dtype=np.float32)
    hq = he.copy()
    rng = np.random.default_rng(seed)
    dh = []
    for l, w in enumerate(plan["w"]):
        ze = he @ w
        step = plan["step"][l]
        if step is None:
            zq = hq @ w
        else:
            qa = np_quant_act(hq, plan["lo"][l], step, plan["lv"][l], rng,
                              stochastic=cfg.stochastic, anti=cfg.anti)
            hh = plan["lo"][l][None, :] + qa * step[None, :]
            zq = hh @ plan["what"][l]
            e = plan["resid"][l]
            if cfg.wmeanfix and e is not None:
                zq = zq + hh.mean(axis=0, dtype=np.float64).astype(np.float32) @ e
        he = np.maximum(ze, 0.0)
        hq = np.maximum(zq, 0.0)
        d = (hq.mean(axis=0, dtype=np.float64) - he.mean(axis=0, dtype=np.float64))
        dh.append(d)
        hq = hq - d.astype(np.float32)
    return dh


def np_forward(x0, plan, cfg, rng, want_layers=False):
    """One quantised forward pass in raw NumPy.  Returns the final activations.

    The arithmetic here is EXACTLY the grouped bit-sliced kernel's: the codes
    are integers, the per-group scale is a power of two, and the weight scale
    factors out of the contraction.  Bit-slicing an integer product is an
    identity, so nothing about the variance measured here depends on the
    packing actually being done.
    """
    h = np.asarray(x0, dtype=np.float32)
    rows = []
    for l, w in enumerate(plan["w"]):
        step = plan["step"][l]
        if step is None:                        # exact layer
            z = h @ w
        else:
            lo, lv = plan["lo"][l], plan["lv"][l]
            qa = np_quant_act(h, lo, step, lv, rng,
                              stochastic=cfg.stochastic, anti=cfg.anti)
            hq = lo[None, :] + qa * step[None, :]
            z = hq @ plan["what"][l]
            e = plan["resid"][l]
            if cfg.wmeanfix and e is not None:
                # hbar @ (W - What): the BATCH MEAN of the weight-quantisation
                # residual, folded back.  A quantised weight is fixed for the
                # whole run, so its error is BIAS and does not divide by N;
                # this cancels its first-order part exactly on this batch and
                # leaves only a per-sample fluctuation, which does.
                z = z + hq.mean(axis=0, dtype=np.float64).astype(np.float32) @ e
        h = np.maximum(z, 0.0)
        if cfg.varmatch and step is not None:
            # VARIANCE MATCHING -- the bias fix that needs no exact reference.
            #
            # relu is convex, so injecting variance v raises E[relu].  But
            # z_q = z + eps is still very nearly Gaussian (z is, by CLT over
            # 256 inputs; eps is, by CLT over 256 independent roundings), so
            # the shift is exactly the difference of two Gaussian relu means
            # at the SAME mean and two different sds:
            #
            #   dh = [m Phi(m/sq) + sq phi(m/sq)] - [m Phi(m/s) + s phi(m/s)]
            #
            # ``s`` is the pilot's; ``sq`` is measured from the scored batch
            # itself -- 2 reductions over the (N,K) array, ~2 FLOPs a sample a
            # neuron -- so NO exact forward pass is needed anywhere.  Nothing
            # is linearised in the perturbation, which matters because the
            # perturbation is ~50% of the signal.
            m, s = plan["m"][l], plan["s"][l]
            vq = np.maximum(z.var(axis=0, dtype=np.float64) - s ** 2, 0.0)
            h = h - (_relu_mean(m, np.sqrt(s ** 2 + vq)) - _relu_mean(m, s)
                     ).astype(np.float32)
        if plan["dh"] is not None:
            h = h - plan["dh"][l].astype(np.float32)
        if want_layers:
            rows.append(h.mean(axis=0, dtype=np.float64))
    return (h, rows) if want_layers else h


def _relu_mean(m, s):
    a = m / s
    return m * _Phi(a) + s * _phi(a)


class Cfg:
    """Precision schedule + switches for one experiment."""

    def __init__(self, ba=4, bw=4, kappa=3.0, groups=6, stochastic=True,
                 stoch_w=True, anti=False, wmeanfix=True, relufix=False,
                 varmatch=False, kappa0=None, ba_sched=None, bw_sched=None,
                 exact_layers=()):
        self.ba, self.bw, self.kappa, self.groups = ba, bw, kappa, groups
        self.stochastic, self.stoch_w = stochastic, stoch_w
        self.anti, self.wmeanfix, self.relufix = anti, wmeanfix, relufix
        self.varmatch = varmatch
        self.kappa0 = kappa0
        self.ba_sched, self.bw_sched = ba_sched, bw_sched
        self.exact = set(exact_layers)

    def ba_at(self, l):
        if l in self.exact:
            return None
        return self.ba if self.ba_sched is None else self.ba_sched[l]

    def bw_at(self, l):
        if l in self.exact:
            return None
        return self.bw if self.bw_sched is None else self.bw_sched[l]

    def label(self):
        s = (f"a{self.ba}w{self.bw} k{self.kappa:g} G{self.groups}"
             if self.ba_sched is None else
             f"a* w{self.bw} k{self.kappa:g} G{self.groups}")
        if self.anti:
            s += " anti"
        if not self.wmeanfix:
            s += " nowfix"
        if not self.stoch_w:
            s += " detW"
        if not self.stochastic:
            s += " RTN"
        if self.relufix:
            s += " lin1fix"
        if self.varmatch:
            s += " vm"
        if self.kappa0:
            s += f" k0={self.kappa0:g}"
        if self.exact:
            s += f" ex{len(self.exact)}"
        return s


# --------------------------------------------------------------------------
# Cost model.  Every constant is measured, not assumed -- see scripts/44
# --------------------------------------------------------------------------


def packed_layer_cost(k_in, k_out, ba, bw, lanes=32):
    """Billed FLOPs per SAMPLE for one bit-sliced layer.

    ``w = ceil(k_in / 32)`` uint32 words hold the packed contraction axis.
    Per output and per ``(p, q)`` bit-plane pair:

        w      bitwise_and        (1 FLOP / uint32 element, 32 lanes)
      + w      bitwise_count      (1 FLOP / uint32 element)
      + (w-1)  add.reduce         (int32 accumulator, rate 1.0)
      = 3w - 1

    plus 2 per pair to shift-and-accumulate ``2^(p+q)``, and the
    quantise+pack of the layer input and the dequantise of its output.
    """
    if ba is None or bw is None:
        return 2.0 * k_in * k_out - k_out
    w = -(-k_in // lanes)
    core = ba * bw * (3 * w - 1) * k_out
    comb = 2 * ba * bw * k_out
    # quantise: (h-lo)*inv_step, +u, floor, clip(2) = 5; pack: per plane an
    # AND (packbits reads nonzero) + packbits = 2, plus the uint8->uint32
    # gather at 5/16 of a word per plane.
    quant = k_in * (5.0 + ba * (2.0 + 5.0 / 16.0))
    rng = k_in                       # one uniform per input element
    deq = 3.0 * k_out                # scale, bias, relu
    return core + comb + quant + rng + deq


def dense_layer_cost(k_in, k_out):
    return 2.0 * k_in * k_out - k_out
