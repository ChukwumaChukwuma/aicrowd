"""Local scoring harness — an exact replica of the whestbench scoring loop,
plus the two measurements whestbench cannot give us.

whestbench reports ``final_layer_mse`` against a reference that is itself a
Monte-Carlo mean.  That number is biased upward by the reference's own
sampling variance, and near the noise floor the bias *is* the signal.  So each
suite carries two independent reference halves ``a`` and ``b``, and this
harness reports

``unbiased_true_mse``
    ``mean_i (p_i - a_i)(p_i - b_i)``.  Unbiased for the true MSE with the
    reference variance removed exactly, because ``E[(δ+ε₁)(δ+ε₂)] = δ²`` when
    ``ε₁ ⟂ ε₂``.  It can and should go negative when the true error is below
    the reference noise; that is information, not an error, and it is never
    clamped here.

``unbiased_true_mse_stderr``
    its standard error.  **The neurons are not independent** — all 256 of them
    are driven by the same input vector — so with ``C = Σ/n_half`` the correct
    variance is

        Var(T) = (1/n²) · [ 2 δᵀ C δ  +  ‖C‖_F² ]

    A diagonal-only version of this formula understates the standard error by
    a measured factor of **4.7×** at this shape, because
    ``‖Σ‖_F² / Σ_j Σ_jj² ≈ 32``.  When a suite carries only the diagonal, the
    harness says so and labels the figure a lower bound rather than quoting it
    as if it were the standard error.

Caveat this harness will not hide: ``final_layer_mse`` measured against a local
suite is *not* the leaderboard number unless the reference has 1e9 samples —
it carries an extra ``v/(2·n_per_half)``.  Only ``unbiased_true_mse +
v/1e9`` is comparable to a leaderboard score, and
:func:`leaderboard_equivalent` is the only function that forms it.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .contract import (
    FLOP_BUDGET,
    GT_SAMPLES_OFFICIAL,
    LAMBDA_FLOPS_PER_SECOND,
    MEASURED_AVG_VARIANCE,
    MULTIPLIER_FLOOR,
    ScoreReport,
    effective_compute,
    score_multiplier,
)
from .suite import Suite

#: A kernel is ``f(weights) -> (depth, width)`` flopscope array.  ``weights``
#: is the list of flopscope float32 matrices.
Kernel = Callable[..., Any]


def run_billed(kernel: Kernel, weights_np: list[np.ndarray], *,
               flop_budget: int = FLOP_BUDGET) -> tuple[np.ndarray, int, float]:
    """Run one kernel inside a real flopscope BudgetContext.

    Returns ``(prediction, flops_used, residual_wall_time_s)``.  The kernel is
    billed exactly as the grader would bill it; nothing is estimated.
    """
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    fw = [fnp.asarray(w) for w in weights_np]
    with flops.BudgetContext(flop_budget=flop_budget, quiet=True) as ctx:
        out = kernel(fw)
    return (
        np.asarray(out, dtype=np.float64),
        int(ctx.flops_used),
        float(ctx.residual_wall_time_s),
    )


def unbiased_mse(pred_final: np.ndarray, a_final: np.ndarray, b_final: np.ndarray,
                 cov: np.ndarray | None, var_diag: np.ndarray,
                 n_half: int) -> tuple[float, float, bool]:
    """Unbiased true MSE and its standard error.

    Returns ``(T, se, se_is_exact)``.  ``se_is_exact`` is False when only the
    diagonal of the reference covariance is available, in which case ``se`` is
    a lower bound (understated by ~4.7x at this shape).
    """
    da = pred_final - a_final
    db = pred_final - b_final
    n = pred_final.shape[0]
    T = float(np.mean(da * db))

    if cov is not None:
        C = cov / n_half
        d = 0.5 * (da + db)
        var = (2.0 * float(d @ (C @ d)) + float(np.sum(C * C))) / (n * n)
        return T, float(np.sqrt(max(var, 0.0))), True

    tau2 = var_diag / n_half
    d2 = da * db
    var = float(np.sum(2.0 * d2 * tau2 + tau2 * tau2)) / (n * n)
    return T, float(np.sqrt(max(var, 0.0))), False


def evaluate(
    kernel: Kernel,
    suite: Suite,
    *,
    name: str = "kernel",
    flop_budget: int = FLOP_BUDGET,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
    seed: int = 0,
    predictions_out: list | None = None,
) -> ScoreReport:
    """Score ``kernel`` on ``suite``, billing every MLP separately.

    FLOP counts are shape-determined and come out identical across MLPs, but
    residual wall time does not (4-10% run-to-run spread), so it is measured
    per MLP and its spread is reported rather than a single draw being reused.
    """
    per_mlp: list[dict[str, Any]] = []
    fl_sum = all_sum = t_sum = 0.0
    t_var_sum = 0.0
    se_exact = True
    flops_list: list[int] = []
    resid_list: list[float] = []
    adj_sum = 0.0

    for i in range(suite.n_mlps):
        w = suite.weights(i)
        pred, fl, resid = run_billed(kernel, w, flop_budget=flop_budget)
        del w
        if predictions_out is not None:
            predictions_out.append(pred.copy())

        C = effective_compute(fl, resid, lam)
        failed = C > flop_budget
        mult = score_multiplier(C, flop_budget, failed=failed)
        pred_eff = np.zeros_like(pred) if failed else pred

        a, b = suite.gt_a[i], suite.gt_b[i]
        g = 0.5 * (a + b)
        fl_mse = float(np.mean((pred_eff[-1] - g[-1]) ** 2))
        al_mse = float(np.mean((pred_eff - g) ** 2))

        cov = suite.final_cov[i] if suite.final_cov is not None else None
        T, se, exact = unbiased_mse(
            pred[-1], a[-1], b[-1], cov, suite.final_var[i], suite.n_per_half
        )
        se_exact &= exact

        fl_sum += fl_mse
        all_sum += al_mse
        t_sum += T
        t_var_sum += se * se
        adj_sum += fl_mse * mult
        flops_list.append(fl)
        resid_list.append(resid)
        per_mlp.append({
            "mlp_seed": suite.mlp_seeds[i],
            "final_layer_mse": fl_mse,
            "all_layers_mse": al_mse,
            "unbiased_true_mse": T,
            "adjusted": fl_mse * mult,
            "flops_used": fl,
            "residual_wall_time_s": resid,
            "multiplier": mult,
            "failed": failed,
        })

    M = suite.n_mlps
    mean_resid = float(np.mean(resid_list))
    C_mean = effective_compute(float(np.mean(flops_list)), mean_resid, lam)
    r = ScoreReport(
        estimator=name,
        n_mlps=M,
        seed=seed,
        adjusted_final_layer_score=adj_sum / M,
        final_layer_mse=fl_sum / M,
        all_layers_mse=all_sum / M,
        compute_ratio=C_mean / flop_budget,
        mean_multiplier=max(MULTIPLIER_FLOOR, C_mean / flop_budget),
        flops_used=float(np.mean(flops_list)),
        residual_wall_time_s=mean_resid,
        per_mlp=per_mlp,
        unbiased_true_mse=t_sum / M,
        unbiased_true_mse_stderr=float(np.sqrt(t_var_sum) / M),
        gt_samples=suite.gt_samples,
    )
    r.notes = (
        f"residual spread {np.std(resid_list) / max(mean_resid, 1e-12):.1%}; "
        + ("stderr exact (full reference covariance used)" if se_exact else
           "STDERR IS A LOWER BOUND: suite carries only the diagonal of the "
           "reference covariance, so cross-neuron terms are missing "
           "(understates by ~4.7x at this shape)")
    )
    return r


def leaderboard_equivalent(r: ScoreReport,
                           avg_variance: float = MEASURED_AVG_VARIANCE) -> tuple[float, float]:
    """Translate a local report into what the leaderboard would show.

    A local suite's raw ``final_layer_mse`` carries its own reference noise,
    which is far larger than the official 1e9-sample reference's.  The
    comparable quantity is the *true* MSE plus the official reference variance:

        raw_equivalent = unbiased_true_mse + v / 1e9

    Returns ``(raw_equivalent, adjusted_equivalent)``.
    """
    raw = float(r.unbiased_true_mse or 0.0) + avg_variance / GT_SAMPLES_OFFICIAL
    return raw, raw * r.mean_multiplier


def report_dict(r: ScoreReport) -> dict[str, Any]:
    """Flatten a report into ledger fields, keeping every mandatory number."""
    raw_eq, adj_eq = leaderboard_equivalent(r)
    return {
        "estimator": r.estimator,
        "n_mlps": r.n_mlps,
        "seed": r.seed,
        "gt_samples": r.gt_samples,
        "adjusted_final_layer_score": r.adjusted_final_layer_score,
        "raw_final_layer_mse": r.final_layer_mse,
        "all_layers_mse": r.all_layers_mse,
        "compute_ratio": r.compute_ratio,
        "mean_multiplier": r.mean_multiplier,
        "flops_used": r.flops_used,
        "residual_wall_time_s": r.residual_wall_time_s,
        "unbiased_true_mse": r.unbiased_true_mse,
        "unbiased_true_mse_stderr": r.unbiased_true_mse_stderr,
        "leaderboard_equivalent_raw_mse": raw_eq,
        "leaderboard_equivalent_adjusted": adj_eq,
        "notes": r.notes,
    }
