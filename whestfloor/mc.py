"""Raw-NumPy Monte Carlo.  Never shipped; used for ground truth and diagnostics.

Kept strictly separate from :mod:`whestfloor.kernels` (which is flopscope-only
and *is* the submission) so that nothing measured here can leak into a graded
path.

Performance notes (measured on this machine, width 256 / depth 32):

===========  ===========  ==============
chunk         all layers   throughput
===========  ===========  ==============
1024          yes          230 GFLOP/s
1024          no           269 GFLOP/s
4096          yes          205 GFLOP/s
8192          yes           95 GFLOP/s
===========  ===========  ==============

The whole loop is memory-bandwidth bound, not FLOP bound: a chunk of 8192×256
float32 is 8 MB and falls out of cache, while 1024×256 is 1 MB and stays in.
Small chunks are therefore *faster*, not slower — the opposite of the usual
BLAS intuition, and worth 2.8×.  Ground truth is the dominant compute consumer
in this project, so this factor is worth more than any algorithmic nicety.

Accuracy: per-chunk sums are taken in float32 (NumPy uses pairwise summation,
so the error is O(log2(chunk)·u·Σ|x|) ≈ 6e-7 relative per chunk) and folded
into a float64 accumulator, giving ~1e-9 relative on the final mean — three
orders below the 1.3e-5 target.  The **final layer**, which is the only one the
score depends on, is accumulated in float64 throughout regardless.
"""

from __future__ import annotations

import numpy as np

from .contract import DEPTH, WIDTH

#: Chunk that keeps the working set in cache.  See the table above.
FAST_CHUNK = 1024


def make_mlp(width: int = WIDTH, depth: int = DEPTH, seed: int = 0) -> list[np.ndarray]:
    """Reproduce ``whestbench.generation.sample_mlp`` exactly."""
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt(2.0 / width))
    return [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]


def layer_means(
    weights: list[np.ndarray],
    n_samples: int,
    seed: int,
    *,
    chunk: int = FAST_CHUNK,
    want_var: bool = True,
    all_layers: bool = True,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Streamed per-layer means, matching whestbench's arithmetic.

    float32 inputs, float32 matmuls, ReLU after every layer, float64
    accumulation of the final layer.  Samples are folded in and dropped, so
    peak memory is two chunk buffers.

    ``all_layers=False`` accumulates only the final layer — 15% faster and all
    the score depends on.  The returned array is then ``(1, width)``.

    Set ``dtype=np.float64`` to run the identical estimate in double precision;
    ``scripts/02_float32_bias.py`` uses that to bound the systematic gap
    between the grader's float32 reference and exact arithmetic.
    """
    depth = len(weights)
    width = weights[0].shape[0]
    Ws = weights if dtype is np.float32 else [w.astype(dtype) for w in weights]
    rng = np.random.default_rng(seed)

    nrows = depth if all_layers else 1
    sums = np.zeros((nrows, width), dtype=np.float64)
    final_sum = np.zeros(width, dtype=np.float64)
    sq = np.zeros(width, dtype=np.float64) if want_var else None

    a = np.empty((chunk, width), dtype=dtype)
    b = np.empty((chunk, width), dtype=dtype)
    zero = dtype(0.0)
    done = 0
    while done < n_samples:
        nb = min(chunk, n_samples - done)
        xa, xb = a[:nb], b[:nb]
        if dtype is np.float32:
            xa[...] = rng.standard_normal((nb, width), dtype=np.float32)
        else:
            xa[...] = rng.standard_normal((nb, width))
        src, dst = xa, xb
        for li in range(depth):
            np.matmul(src, Ws[li], out=dst)
            np.maximum(dst, zero, out=dst)
            if all_layers and li < depth - 1:
                sums[li] += dst.sum(axis=0, dtype=dtype)
            src, dst = dst, src
        # src now holds the final-layer activations; accumulate in float64.
        final_sum += src.sum(axis=0, dtype=np.float64)
        if sq is not None:
            sq += np.einsum("ij,ij->j", src, src, dtype=np.float64)
        done += nb

    sums[-1] = final_sum
    means = sums / done
    var = (sq / done - means[-1] ** 2) if sq is not None else None
    return means, var
