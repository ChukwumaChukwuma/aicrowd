"""Weight derivation for the *official* whestbench datasets (seed protocol 3.0).

This is deliberately separate from :func:`whestfloor.mc.make_mlp`.  The two
disagree, and the difference is not cosmetic:

* ``whestfloor.mc.make_mlp(w, d, s)`` seeds with ``default_rng(s)`` — a valid
  draw from the same distribution, which is all a locally generated suite needs.
* The official bake seeds with ``default_rng(SeedSequence(s).spawn(3)[0])``.
  A spawned ``SeedSequence`` carries ``spawn_key=(0,)``, so it hashes to a
  completely different PCG64 state than the bare integer does.  Feeding a local
  ``make_mlp`` network against official ground truth would score ~``v`` (0.05),
  not ~1e-10.

Source of truth: ``whestbench/src/whestbench/seeds.py:derive_seed_streams`` and
``whestbench/src/whestbench/generation.py:sample_mlp``::

    weight_ss, sample_ss, estimator_ss = SeedSequence(int(mlp_seed)).spawn(3)
    rng   = default_rng(weight_ss)
    scale = float(sqrt(2.0 / width))
    W_l   = (rng.standard_normal((width, width)) * scale).astype(float32)

whestbench calls these through ``flopscope.numpy``; ``fnp.random.SeedSequence``
*is* ``numpy.random.SeedSequence`` and ``fnp.random.default_rng`` produces a
bit-identical stream, both checked in ``scripts/19_fetch_official_suite.py``.
The draw is float64, scaled in float64, and only then cast to float32 — order
matters, and this reproduces it.
"""

from __future__ import annotations

import numpy as np

from .contract import DEPTH, WIDTH

#: ``whestbench.dataset_io.SEED_PROTOCOL_VERSION_V3``
SEED_PROTOCOL = "whestbench_explicit_per_mlp_seeds/3.0"


def derive_seed_streams(mlp_seed: int) -> tuple[np.random.SeedSequence,
                                                np.random.SeedSequence, int]:
    """Return ``(weight_ss, sample_ss, estimator_seed)`` — mirrors whestbench."""
    weight_ss, sample_ss, estimator_ss = np.random.SeedSequence(
        int(mlp_seed)).spawn(3)
    return weight_ss, sample_ss, int(estimator_ss.generate_state(1)[0])


def derive_estimator_seed(mlp_seed: int) -> int:
    """The protocol-3.0 ``MLP.seed`` handed to the participant's estimator."""
    return derive_seed_streams(mlp_seed)[2]


def make_official_mlp(width: int = WIDTH, depth: int = DEPTH,
                      mlp_seed: int = 0) -> list[np.ndarray]:
    """Reproduce the official dataset's weight matrices for one ``mlp_seed``."""
    weight_ss, _, _ = derive_seed_streams(mlp_seed)
    rng = np.random.default_rng(weight_ss)
    scale = float(np.sqrt(2.0 / width))
    return [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
