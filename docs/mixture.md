# The Gaussian-mixture route: what the form can reach, and why it stops there

**Result. TODO-HEADLINE**

Reproducible from `scripts/47_mixture.py` (`--mode ceiling|anatomy|propagate`);
machinery in `whestfloor/mixture.py`; ledger rows via
`scripts/48_record_mixture.py`. `submission/` untouched.

---

## 0. The hypothesis, and why it deserved a test

Two routes to the top of the leaderboard are now closed with bounds.
`docs/cost_floor.md`: no sampler in the family reaches dpskv5's raw 3.6e-9 at
its billed 2.43e10 FLOPs — a sampler at that budget scores `v/N = 7.8e-6`, 2,160x
worse. `docs/traj_closure.md`: no moment closure reaches it either — the
trajectory-calibrated chain is `r = 2.0` deployable and `r ~ 5-7` with *exact*
cumulants, against the ~36x on MSE that even perfect cumulants buy.

What is left is a deterministic method costing a handful of covariance
propagations. A **K-component Gaussian mixture** is exactly that shape: each
component is propagated by the exact rectified-Gaussian machinery
(`whestfloor/relu_moments.py`, Mehler), so there is no *moment closure* per
component, and the cost is `K` propagations.

## 1. The form is its own ceiling

Whatever produces the components, a mixture's answer is

```
E[relu(z^32_j)]  ~=  sum_k w_k relu_mean(m_kj, s_kj).                       (*)
```

So the family can be bounded without building anything: take the **exact** law
of `z^32` from Monte Carlo, cut it into `K` cells along a chosen `r`-dimensional
frame, and evaluate `(*)` with the **true** per-cell conditional mean and
standard deviation. No propagation error, no closure error, no fitting are
charged. That is a ceiling on every mixture whose components are indexed by
those `r` coordinates and Gaussian in the complement — which is precisely the
construction under test.

TODO-CEILING-TABLE

## 2. Why it stops there

TODO-MECHANISM

## 3. The deployable propagator

TODO-PROPAGATE

## 4. Bars, fixed before each run

TODO-BARS
