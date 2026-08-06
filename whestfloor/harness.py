"""Local scoring harness — an exact replica of the whestbench scoring loop,
plus the two measurements whestbench cannot give us.

whestbench reports ``final_layer_mse`` against a reference that is itself a
Monte-Carlo mean.  That number is biased upward by the reference's own
sampling variance, and near the noise floor the bias *is* the signal.  So this
harness additionally reports

``unbiased_true_mse``
    ``mean_i (p_i - a_i)(p_i - b_i)`` over two independent reference halves
    ``a`` and ``b``.  Unbiased for the true MSE with the reference variance
    removed exactly, because ``E[(δ+ε₁)(δ+ε₂)] = δ²`` when ``ε₁ ⟂ ε₂``.

``unbiased_true_mse_stderr``
    its standard error, computed from the measured per-neuron activation
    variance.  Without this a claim of "at the floor" is unfalsifiable.

The whestbench-identical number is still reported, always, as
``final_layer_mse``.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .contract import (
    FLOP_BUDGET,
    LAMBDA_FLOPS_PER_SECOND,
    ScoreReport,
    effective_compute,
    score_multiplier,
)
from .suite import Suite

#: A kernel is ``f(weights, xp) -> (depth, width) array``.
Kernel = Callable[[list, Any], Any]


def measure_flops(kernel: Kernel, weights: list, *, flop_budget: int = FLOP_BUDGET
                  ) -> tuple[int, float, float]:
    """Run the kernel once under a real flopscope BudgetContext.

    Returns ``(flops_used, residual_wall_time_s, backend_time_s)``.  The FLOP
    count is shape-determined, so one MLP prices the whole suite — but the
    residual wall time is machine-dependent and is reported, never assumed.
    """
    import flopscope as flops  # noqa: PLC0415

    from .backend import FlopscopeBackend  # noqa: PLC0415

    xp = FlopscopeBackend()
    fw = [xp.asarray(w) for w in weights]
    with flops.BudgetContext(flop_budget=flop_budget, quiet=True) as ctx:
        out = kernel(fw, xp)
        _ = xp.to_numpy(out)
    return (
        int(ctx.flops_used),
        float(ctx.residual_wall_time_s),
        float(ctx.flopscope_backend_time_s),
    )


def evaluate(
    kernel: Kernel,
    suite: Suite,
    *,
    name: str = "kernel",
    flop_budget: int = FLOP_BUDGET,
    flops_used: int | None = None,
    residual_wall_time_s: float = 0.0,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
    bill_first_mlp: bool = True,
    seed: int = 0,
    predictions_out: list | None = None,
) -> ScoreReport:
    """Score ``kernel`` on ``suite``.

    Predictions are produced on the NumPy backend for speed; the FLOP count
    comes from one flopscope-billed pass over the first MLP unless supplied.
    """
    from .backend import NumpyBackend  # noqa: PLC0415

    xp = NumpyBackend()

    if flops_used is None and bill_first_mlp:
        flops_used, residual_wall_time_s, _ = measure_flops(
            kernel, suite.weights(0), flop_budget=flop_budget
        )
    if flops_used is None:
        raise ValueError("flops_used must be supplied when bill_first_mlp is False")

    C = effective_compute(flops_used, residual_wall_time_s, lam)
    failed = C > flop_budget
    mult = score_multiplier(C, flop_budget, failed=failed)

    per_mlp: list[dict[str, Any]] = []
    t_sum = 0.0
    t_var_sum = 0.0
    fl_sum = 0.0
    all_sum = 0.0

    for i in range(suite.n_mlps):
        w = suite.weights(i)
        t0 = time.perf_counter()
        pred = np.asarray(xp.to_numpy(kernel(w, xp)), dtype=np.float64)
        dt = time.perf_counter() - t0
        del w
        if predictions_out is not None:
            predictions_out.append(pred.copy())

        a = suite.gt_a[i]
        b = suite.gt_b[i]
        g = 0.5 * (a + b)

        if failed:
            pred_eff = np.zeros_like(pred)
        else:
            pred_eff = pred

        fl_mse = float(np.mean((pred_eff[-1] - g[-1]) ** 2))
        al_mse = float(np.mean((pred_eff - g) ** 2))

        da = pred[-1] - a[-1]
        db = pred[-1] - b[-1]
        t_m = float(np.mean(da * db))

        # Var(T_m) = (1/n^2) sum_i [2 d_i^2 tau^2 + tau^4], tau^2 = v_i/n_half
        tau2 = suite.final_var[i] / suite.n_per_half
        d2 = np.maximum(da * db, 0.0)
        n = pred.shape[1]
        t_var = float(np.sum(2.0 * d2 * tau2 + tau2 * tau2) / (n * n))

        fl_sum += fl_mse
        all_sum += al_mse
        t_sum += t_m
        t_var_sum += t_var
        per_mlp.append(
            {
                "mlp_seed": suite.mlp_seeds[i],
                "final_layer_mse": fl_mse,
                "all_layers_mse": al_mse,
                "unbiased_true_mse": t_m,
                "adjusted": fl_mse * mult,
                "predict_wall_s": dt,
            }
        )

    M = suite.n_mlps
    final_layer_mse = fl_sum / M
    all_layers_mse = all_sum / M
    unbiased = t_sum / M
    unbiased_se = float(np.sqrt(t_var_sum) / M)

    return ScoreReport(
        estimator=name,
        n_mlps=M,
        seed=seed,
        adjusted_final_layer_score=final_layer_mse * mult,
        final_layer_mse=final_layer_mse,
        all_layers_mse=all_layers_mse,
        compute_ratio=C / flop_budget,
        mean_multiplier=mult,
        flops_used=float(flops_used),
        residual_wall_time_s=residual_wall_time_s,
        per_mlp=per_mlp,
        unbiased_true_mse=unbiased,
        unbiased_true_mse_stderr=unbiased_se,
        gt_samples=suite.gt_samples,
    )


def report_dict(r: ScoreReport) -> dict[str, Any]:
    """Flatten a report into ledger fields, keeping every mandatory number."""
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
        "notes": r.notes,
    }
