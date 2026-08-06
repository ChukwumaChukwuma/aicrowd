"""FROZEN CONTRACT for the ARC White-Box Estimation Challenge.

This module is the single owner of every constant, type and signature that the
rest of the repository depends on.  Nothing here may change without bumping
``CONTRACT_VERSION`` and recording the bump in ``ledger/`` — a silent change to
any of these numbers invalidates every measurement taken before it.

Every constant below is sourced from code, not prose.  The provenance comment
on each line names the file it was read from so the claim is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

CONTRACT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Problem shape.  whestbench/src/whestbench/cli.py:394,457 (competition spec).
# ---------------------------------------------------------------------------
WIDTH = 256
DEPTH = 32

#: He initialisation: W[i, j] ~ N(0, 2 / width), float32.
#: whestbench/src/whestbench/generation.py:sample_mlp
WEIGHT_STD = (2.0 / WIDTH) ** 0.5

#: Forward pass is ``x @ W`` for a batch of row vectors; W is (in, out).
#: whestbench/src/whestbench/simulation.py:run_mlp

# ---------------------------------------------------------------------------
# Compute model.  whestbench/src/whestbench/budget.py
# ---------------------------------------------------------------------------
FLOP_BUDGET = 272_000_000_000  # cli.py:129,394,457 -> 2.72e11
LAMBDA_FLOPS_PER_SECOND = 1.0e11  # budget.py:LAMBDA_FLOPS_PER_SECOND
MULTIPLIER_FLOOR = 0.1  # budget.py:score_multiplier

#: Effective compute below which the multiplier is clamped at the floor.  Any
#: FLOPs spent under this line are *free* in score terms.  This is the single
#: most important number in the competition: 2.72e10.
FREE_COMPUTE = int(MULTIPLIER_FLOOR * FLOP_BUDGET)

#: Residual (participant-Python) wall time that alone would consume the whole
#: free allowance, in seconds.  0.272 s.  flopscope backend time and flopscope
#: dispatch overhead are NOT residual; only our own Python is.
FREE_RESIDUAL_SECONDS = FREE_COMPUTE / LAMBDA_FLOPS_PER_SECOND

# Hard limits (whestbench/src/whestbench/scoring.py:ContestSpec defaults).
SETUP_TIMEOUT_S = 5.0
PREDICT_TIMEOUT_S = 30.0
WALL_TIME_LIMIT_S = 60.0
MEMORY_LIMIT_MB = 65_536

# ---------------------------------------------------------------------------
# Ground truth.  whestbench/src/whestbench/simulation.py:sample_layer_statistics
# ---------------------------------------------------------------------------
#: Official baked datasets use N = 1e9 Monte-Carlo forward passes per MLP.
GT_SAMPLES_OFFICIAL = 1_000_000_000

#: Ground-truth chunk size at width 256: max(1024, min(16384, 2**20 // 256)).
GT_CHUNK_SIZE = 4096

#: Mean per-neuron final-layer activation variance.
#:
#: The challenge documentation quotes 0.18 and derives a ~2e-10 floor from it.
#: That number is WRONG FOR THIS SHAPE.  Measured here (scripts/02_adv_variance.py,
#: 128 MLPs x 8192 samples, confirmed by an independent second run):
#:
#:     depth  4 : 0.3292 +- 0.0035
#:     depth  8 : 0.1885 +- 0.0048   <-- this is where 0.18 comes from
#:     depth 16 : 0.0999 +- 0.0035
#:     depth 32 : 0.0551 +- 0.0023   <-- the phase-1 competition shape
#:
#: 0.18 is the depth-8 warm-up value, inherited across the depth change; every
#: dataset-bake example in the whestbench docs still uses --depth 8.  An
#: independent analytic route agrees: v = q*(1-c) where c is the two-input
#: overlap under the ReLU arc-cosine map, giving 1-c_32 = 0.0253 as an
#: infinite-width lower bound, approached monotonically from above as width
#: grows (0.084/0.079/0.060/0.040 at width 64/128/256/512).
#:
#: This matters directly: it moves the floor down by 3.3x and tightens the
#: per-neuron accuracy target by 1.81x.
MEASURED_AVG_VARIANCE = 0.0551
MEASURED_AVG_VARIANCE_SE = 0.0023
DOC_QUOTED_AVG_VARIANCE = 0.18  # kept only to document the discrepancy

#: v varies strongly across MLPs: sd 0.0265 over 128 draws, CV 0.48, range
#: [0.0144, 0.1547].  The floor is a per-suite random variable, not a
#: constant; over a 100-MLP suite it is 5.50e-11 +- 0.26e-11 (4.8%).
AVG_VARIANCE_CV = 0.48


def gt_noise_floor(avg_variance: float = MEASURED_AVG_VARIANCE,
                   n_samples: int = GT_SAMPLES_OFFICIAL) -> float:
    """Raw final-layer MSE that a *perfect* estimator still incurs.

    The reference is itself a Monte-Carlo mean of ``n_samples`` draws, so it
    carries variance ``avg_variance / n_samples`` per neuron.  Measured error
    decomposes as ``true_error + reference_variance``; with zero true error the
    measured MSE equals the reference variance.  Nobody can score below this.
    """
    return avg_variance / n_samples


#: Raw MSE floor 5.50e-11; adjusted (at the 0.1 multiplier) 5.50e-12.
#: Per-neuron RMS a perfect estimator would still show: 7.42e-6.
RAW_MSE_FLOOR = gt_noise_floor()
ADJUSTED_FLOOR = RAW_MSE_FLOOR * MULTIPLIER_FLOOR
TARGET_RMS = RAW_MSE_FLOOR ** 0.5

#: Errors injected by successive layers add incoherently (measured: covariance
#: propagation injects ~2.0e-3 per layer and lands at ~9.2e-3 end to end, and
#: sqrt(32)*2.0e-3 = 1.13e-2).  So the per-layer modelling budget is:
PER_LAYER_BAR = TARGET_RMS / (DEPTH ** 0.5)


# ---------------------------------------------------------------------------
# Cost of a Monte-Carlo forward pass, used to price sampling-based methods.
# Derived, not assumed; verified against flopscope in scripts/01.
# ---------------------------------------------------------------------------
def forward_pass_flops(width: int = WIDTH, depth: int = DEPTH) -> int:
    """Analytic flopscope cost of one N(0,1) sample through the whole net.

    Per layer: a (1, width) @ (width, width) matmul costs ``width * (2*width-1)``
    and the ReLU (``fnp.maximum``) costs ``width``.  The float32 standard-normal
    draw costs 16 per element.
    """
    per_layer = width * (2 * width - 1) + width
    return 16 * width + depth * per_layer


MC_SAMPLES_AT_BUDGET = FLOP_BUDGET // forward_pass_flops()
MC_SAMPLES_FREE = FREE_COMPUTE // forward_pass_flops()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLPSpec:
    """Everything an estimator is handed about one network."""

    width: int
    depth: int
    weights: Sequence[object]  # depth arrays of shape (width, width), float32
    seed: int = 0


@runtime_checkable
class WhestEstimator(Protocol):
    """The frozen estimator interface.

    Implementations must be pure with respect to ``predict``: no state may
    leak between MLPs except read-only precomputed tables built in ``setup``.
    """

    name: str

    def predict(self, mlp: MLPSpec, budget: int):  # -> fnp.ndarray (depth, width)
        ...


@dataclass
class ScoreReport:
    """Result of scoring one estimator against one suite.

    ``adjusted_final_layer_score`` is never reported without ``raw`` fields —
    the ledger refuses records that omit them.
    """

    estimator: str
    n_mlps: int
    seed: int
    adjusted_final_layer_score: float
    final_layer_mse: float
    all_layers_mse: float
    compute_ratio: float
    mean_multiplier: float
    flops_used: float
    residual_wall_time_s: float
    per_mlp: list = field(default_factory=list)
    unbiased_true_mse: float | None = None
    unbiased_true_mse_stderr: float | None = None
    gt_samples: int | None = None
    notes: str = ""


def score_multiplier(effective_compute: float, flop_budget: int = FLOP_BUDGET,
                     *, failed: bool = False) -> float:
    """Exact replica of whestbench.budget.score_multiplier."""
    if failed or flop_budget <= 0:
        return 1.0
    return max(MULTIPLIER_FLOOR, float(effective_compute) / float(flop_budget))


def effective_compute(flops_used: float, residual_wall_time_s: float,
                      lam: float = LAMBDA_FLOPS_PER_SECOND) -> float:
    """Exact replica of whestbench.budget.effective_compute: C = F + lambda*R."""
    return float(flops_used) + lam * float(residual_wall_time_s)
