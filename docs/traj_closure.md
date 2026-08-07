# The closure-accuracy scalar `r`: baselines, and the ceiling of every per-layer correction

*`docs/integrable_cv.md` reduced the deep control-variate programme to one
number. This page measures it — for what the repository already has, for the
trajectory-calibrated chain of AIcrowd forum topic 18097, and for the oracle
that bounds both.*

**Method credit.** The trajectory-calibrated moment chain is
**@jamesrahenry**'s (AIcrowd discourse topic 18097, "Stabilizing cumulant
propagation at depth 32", graded submission #314695; MIT replication repository
[`jamesrahenry/arc-whitebox-replication`](https://github.com/jamesrahenry/arc-whitebox-replication)).
His fitting targets and ground-truth cumulant tests came from
**@keenanpepper**'s two public datasets,
`keenanpepper/arc-whestbench-higher-moments-2026` and
`keenanpepper/whest-k3-tensors-2026`. Everything in §3 below is his method
transplanted; §2 and §4 are his sec 1 diagnostic re-run on our networks.

---

## 0. The yardstick, fixed before any run

`docs/integrable_cv.md` §3.2: a linear control variate in `relu(z^L)` breaks
even against the shipped basis at

```
r  :=  rms(mu_gauss^L - mu_true^L) / rms(mu_new^L - mu_true^L),   mu^L = E[relu(z^L)]
```

`r = 4.2-4.6` at every depth from 6 to 32, and past it the payoff is steep
(1.9x at `r = 4.4`, 5.4x at `r = 10`, 25x at `r = 30`). So:

> **Pre-registered bar: `r > 4.4` at `L = 8, 16, 24, 32`.**

Reference: 2,000,000 samples per MLP in **two independent halves**, so
`E[(mt - mA)(mt - mB)] = (mt - m_pop)^2` and no figure on this page is inflated
by the reference's own noise (which is `2.4e-04` rms per half-pair, against
closure errors of `5e-03` to `8e-03`). Networks are 24 locally generated MLPs,
seeds `960000+`, disjoint from every other page in this repository and from the
official suite. Reproducible from `scripts/43_traj_closure.py`
(`--mode ref|aux|baseline|ceiling|fit`); machinery in
`whestfloor/trajclosure.py`.

## 1. The baselines reproduce

`--mode baseline`, 12 MLPs. `r` at four depths:

| arm | rms `mu - mu_true` at L=8 / 16 / 24 / 32 | `r` at L=8 / 16 / 24 / 32 |
|---|---|---|
| **Gaussian closure** (the incumbent) | 5.59e-03 / 7.41e-03 / 7.86e-03 / 7.11e-03 | **1.00 / 1.00 / 1.00 / 1.00** |
| all-coincident `(3)`,`(4)` diagrams only | 4.80e-03 / 6.63e-03 / 7.13e-03 / 6.54e-03 | 1.17 / 1.12 / 1.10 / 1.09 |
| **`kappa_3` star** (`cov_prop_edgeworth`) | 3.98e-03 / 5.46e-03 / 5.95e-03 / 5.73e-03 | **1.41 / 1.36 / 1.32 / 1.24** |
| `kappa_3` complete tree catalogue | 4.11e-03 / 5.67e-03 / 6.12e-03 / 5.89e-03 | 1.36 / 1.31 / 1.28 / 1.21 |
| `kappa_3`+`kappa_4` complete tree | 3.94e-03 / 5.53e-03 / 5.99e-03 / 5.78e-03 | 1.42 / 1.34 / 1.31 / 1.23 |

Two things to read off.

**The yardstick is the one the payoff curve was priced against.** The Gaussian
closure is `r = 1` by construction and its absolute rms (7.1e-03 at `L = 32`) is
within 12% of the 6.4e-03 `docs/integrable_cv.md` §6 measured on the official
pair. The `kappa_3` star arm lands at **1.41** against that page's published
**1.45**.

**The complete diagram catalogue is not better than the incomplete one.** The
tree catalogue — every injectivity correction, both Ursell terms, `kappa_4`,
i.e. the object `docs/cumulant_expansion.md` §5 spends nine sections building
and verifies against Monte Carlo to within one standard error — scores
**1.42/1.34/1.31/1.23** against the star's **1.41/1.36/1.32/1.24**. Getting the
source diagrams exactly right buys nothing. §2 says why.

## 2. Where the error actually is, and it is not the rectifier

Decomposing the layer-`L` mean residual of the *uncorrected* chain (one MLP,
150,000-sample cumulants, `--mode aux`): regress `mu_true - mu_closure` on

* `A` — a rich per-neuron basis, `sigma phi(alpha) alpha^k` for `k <= 7` plus
  `sigma Phi`, `sigma`, `1`;
* `A` + the **true** `sigma` and mean errors of the state (`phi(alpha) dsigma`,
  `Phi(alpha) dm`);
* `A` + the **true** `gamma_3` / `gamma_4` of that layer.

| `L` | rms residual | rel. `sigma` error | `R^2` on `A` | `A` + true `dsigma, dm` | `A` + true `gamma_3` | `A` + true `gamma_4` | all |
|---|---|---|---|---|---|---|---|
| 2 | 1.65e-03 | 0.18% | 0.658 | 0.662 | **0.886** | 0.660 | 0.890 |
| 4 | 3.47e-03 | 0.95% | 0.613 | 0.817 | 0.674 | 0.622 | 0.895 |
| 8 | 4.56e-03 | 2.7% | 0.556 | **0.899** | 0.584 | 0.557 | 0.956 |
| 16 | 5.77e-03 | 8.1% | 0.471 | **0.961** | 0.478 | 0.472 | 0.986 |
| 24 | 4.59e-03 | 9.1% | 0.381 | **0.944** | 0.420 | 0.381 | 0.989 |
| 32 | 4.10e-03 | 11.5% | 0.589 | **0.978** | 0.593 | 0.593 | 0.994 |

**Past layer 4 the residual is state drift, not non-Gaussianity.** Knowing the
true cumulants of the layer adds 1-4 points; knowing the state's own `sigma`
error adds 35-50 and takes the fit to 94-98%. This is
`docs/cumulant_expansion.md` §9.4's "the error is in the propagated moments,
not in the last rectification", measured from the other side, and it is why the
complete diagram catalogue is worth nothing in §1.

It also prices the cumulant-knowledge route directly. The analytic per-neuron
`gamma_3` correlates with the true one at **0.83 / 0.69 / 0.32 / 0.35 / 0.29**
at `L = 4 / 8 / 16 / 24 / 32` (star; the tree catalogue is not better), and its
magnitude is 6-7x too small — the transport deficit of
`docs/cumulant_expansion.md` §9.6, confirmed. But since the true `gamma_3` is
worth only 1-4 points of the residual at depth, closing that gap is not the
lever either.

One thing that *is* worth knowing for any future scheme: **the true per-neuron
`gamma_3` field is 84-87% a smooth function of `alpha` alone**, at every depth
(degree-5 polynomial, per layer). So the cumulant field is not the hard part;
it is nearly free from the closure's own state.

## 3. The trajectory-calibrated closure, probe-free

`--mode fit`. jamesrahenry's construction: per-layer linear corrections to the
post-ReLU mean (and optionally the variance and the covariance off-diagonals),
**fitted sequentially on the chain's own rolled-forward state** — at layer `l`
every training net is advanced with the corrections of layers `0..l-1` already
applied, the design matrix is built at the state actually visited, and one
pooled least squares gives layer `l`'s coefficients. Training distribution =
deployment distribution, which is the whole of DAgger.

**One deliberate departure: no probe.** Five of his eight mean features, three
of six variance features and both off-diagonal features are built from an
`N = 4096` plain-MC probe of the target network's own per-layer cumulants
(~6% of his FLOP budget). *A probe-fed mean is worth nothing as a
control-variate mean.* `docs/integrable_cv.md` §3.3: a control variate on `h^L`
with a claimed mean is exactly a shrinkage between the analytic mean and the
sample mean of that layer's activations, so it pays only where the analytic
error beats the noise of the `N = 27,000`-sample pass **that is already
running**. A mean estimated from `N_p` fresh samples contributes `V/N_p` to the
control variate's variance in full, so the device is a loss unless
`N_p >> N` — at which point the probe, not the closure, is the estimator. Every
cumulant here is therefore analytic (star or tree diagrams).

7 train MLPs / 2 held out, first cut:

| feature pack | `r` at L=8 / 16 / 24 / 32 |
|---|---|
| `m3` — mean only, 3 shape features | 1.58 / 1.65 / 1.98 / 1.62 |
| `m8` — mean only, his 8 with analytic cumulants | **1.89 / 1.84 / 1.99 / 1.64** |
| `m12` — mean only, 12 features | 1.88 / 1.81 / 1.98 / 1.58 |
| `freem` — mean (8) + variance (6) | 1.85 / 1.75 / 1.91 / 1.61 |

Real, and better than the `kappa_3` arm's 1.41/1.36/1.32/1.24 — but a long way
from 4.4, and **adding features does not move it**. §4 says why, and the answer
is not the feature map.

## 4. The ceiling: what a PERFECT per-layer correction reaches

`--mode ceiling`, 12 MLPs. Overwrite part of the state with Monte-Carlo truth
at every layer boundary and read the closure's own mean. A fitted correction
cannot beat the oracle it is approximating, so these rows bound every
parameterisation, every feature map and every probe size at once.

| oracle | rms at L=8 / 16 / 24 / 32 | `r` at L=8 / 16 / 24 / 32 |
|---|---|---|
| none (the Gaussian chain) | 5.59e-03 / 7.41e-03 / 7.86e-03 / 7.11e-03 | 1.00 / 1.00 / 1.00 / 1.00 |
| **`E[relu(z^l)] <- truth` every layer** | 2.04e-03 / 1.61e-03 / 1.43e-03 / 1.22e-03 | **2.74 / 4.61 / 5.50 / 5.85** |
| `Var(relu(z^l)) <- truth` every layer | 9.28e-03 / 1.80e-02 / 2.25e-02 / 2.48e-02 | **0.60 / 0.41 / 0.35 / 0.29** |
| both | 4.29e-03 / 5.80e-03 / 6.16e-03 / 5.78e-03 | 1.30 / 1.28 / 1.28 / 1.23 |

Three things, and they decide the direction.

**A perfect mean correction is worth `r = 2.74` at `L = 8` and `5.85` at
`L = 32`.** At `L = 8` that is *below* the 4.4 break-even and at `L = 16` it is
level with it. Only `L = 24` and `L = 32` have oracle headroom at all.

**Re-anchoring the variance to truth makes the chain three times worse.** This
is jamesrahenry's sec 1(b) result reproduced on our networks and from an
independently written chain: his "variances only" reset is 1.8e-4 against a
plain 6.0e-5; ours is `r = 0.29` at `L = 32`. And fixing mean *and* variance
together (1.23) is four times worse than fixing the mean alone (5.85). **The
covariance error is what compensates the mean error downstream**, so any
correction that improves the covariance in isolation is a liability, not a
gain — which is exactly why the `freem` pack above is no better than `m8`.

**The two independent measurements of the ceiling agree.** His mean-only reset
on the official nets is 6.0e-5 -> 1.64e-6, i.e. `r = 6.05`; ours is **5.85**.
Different networks, different closure implementation, same number.

## 5. Status

| # | bar, fixed before the run | measured | verdict |
|---|---|---|---|
| 1 | the Gaussian closure reproduces at `r = 1` and the `kappa_3` arm at the published 1.45 | 1.000 and **1.41** at `L = 8` | **PASS** |
| 2 | a perfect per-layer mean correction clears `r > 4.4` at every one of `L = 8,16,24,32` | **2.74** / 4.61 / 5.50 / 5.85 — fails at `L = 8`, level at `L = 16` | **FAIL** |
| 3 | the probe-free trajectory-calibrated closure clears `r > 4.4` | 1.89 / 1.84 / 1.99 / 1.64 | **FAIL** (§6 continues) |

Bar 2 is the one that matters: it is not a statement about a feature map, it is
a statement about the family. Work continues on the covariance-correction arm
(§6, in progress) because `L = 24` and `L = 32` do have oracle headroom, and on
pricing what `r = 5.85` would actually be worth if it were reached.
