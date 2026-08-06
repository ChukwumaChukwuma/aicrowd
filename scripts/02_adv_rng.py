#!/usr/bin/env python
"""ADVERSARIAL CHECK 4 -- does whestfloor reproduce whestbench's MLPs and GT?

Two separate claims are tested:

  4a  ``mc.make_mlp`` docstring: "Reproduce whestbench.generation.sample_mlp
      exactly."  Tested bit-for-bit against the real whestbench function.

  4b  ``mc.layer_means`` docstring: "matching whestbench's arithmetic".  Tested
      against the real ``whestbench.simulation.sample_layer_statistics``, same
      MLP, same sampling seed, bit-for-bit where possible.

whestbench's package ``__init__`` pulls in HuggingFace ``datasets``/``yaml``/
``jinja2`` which are not installed, so the package object is constructed by
hand with the right ``__path__`` and the leaf modules are imported directly.
No whestbench source is modified.

Run:
    . <prefix>/runenv.sh && OPENBLAS_NUM_THREADS=1 $PY scripts/02_adv_rng.py
"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WHESTBENCH_SRC = os.environ.get(
    "WHESTBENCH_SRC",
    "/tmp/claude-0/-home-user-aicrowd/373914b3-787b-5580-866d-92a576ded1af/"
    "scratchpad/whestbench/src/whestbench",
)

_pkg = types.ModuleType("whestbench")
_pkg.__path__ = [WHESTBENCH_SRC]
sys.modules["whestbench"] = _pkg

import flopscope.numpy as fnp  # noqa: E402
from whestbench.generation import sample_mlp  # noqa: E402
from whestbench.seeds import derive_seed_streams  # noqa: E402
from whestbench.simulation import (  # noqa: E402
    _pick_chunk_size,
    sample_layer_statistics,
)

from whestfloor.mc import layer_means, make_mlp  # noqa: E402


def to_np(a):
    return np.asarray(fnp.to_numpy(a) if hasattr(fnp, "to_numpy") else a)


def check_4a() -> bool:
    print("=== 4a: make_mlp vs whestbench.generation.sample_mlp ===")
    ok_all = True

    print("\n  (i) same generator object (default_rng(seed)) -- the docstring's claim")
    for width, depth, seed in ((64, 4, 0), (64, 4, 7), (256, 8, 3)):
        rng = fnp.random.default_rng(seed)
        wb = sample_mlp(width, depth, rng, seed=seed)
        wb_w = [np.asarray(w) for w in wb.weights]
        ours = make_mlp(width, depth, seed)
        same = all(np.array_equal(a, b) for a, b in zip(wb_w, ours))
        dtypes = {str(a.dtype) for a in wb_w} | {str(a.dtype) for a in ours}
        print(f"    width={width} depth={depth} seed={seed}: bit-identical={same} "
              f"dtypes={sorted(dtypes)}")
        ok_all &= same

    print("\n  (ii) how whestbench ACTUALLY seeds it when baking a dataset")
    print("       dataset.py:184-186 -> weight_ss = SeedSequence(input_seed).spawn(3)[0]")
    print("                             sample_mlp(w, d, default_rng(weight_ss))")
    for input_seed in (0, 1, 42):
        weight_ss, _sample_ss, est_seed = derive_seed_streams(input_seed)
        wb = sample_mlp(64, 4, fnp.random.default_rng(weight_ss), seed=est_seed)
        wb_w = [np.asarray(w) for w in wb.weights]
        ours = make_mlp(64, 4, input_seed)
        same = all(np.array_equal(a, b) for a, b in zip(wb_w, ours))
        corr = float(np.corrcoef(wb_w[0].ravel(), ours[0].ravel())[0, 1])
        print(f"    input_seed={input_seed}: make_mlp(seed) == dataset MLP? {same}  "
              f"(corr of W0 = {corr:+.4f})")
        if same:
            print("      ^ unexpected")

    print("\n  (iii) is the ENSEMBLE the same law?  (this is what actually matters)")
    a = np.concatenate([make_mlp(64, 2, 10_000 + i)[0].ravel() for i in range(20)])
    b = []
    for i in range(20):
        wss, _s, es = derive_seed_streams(20_000 + i)
        m = sample_mlp(64, 2, fnp.random.default_rng(wss), seed=es)
        b.append(np.asarray(m.weights[0]).ravel())
    b = np.concatenate(b)
    scale = np.sqrt(2.0 / 64)
    print(f"    make_mlp   : n={a.size}  mean={a.mean():+.3e}  std={a.std():.8f}")
    print(f"    whestbench : n={b.size}  mean={b.mean():+.3e}  std={b.std():.8f}")
    print(f"    target std = sqrt(2/width) = {scale:.8f}")
    print(f"    std ratio  = {a.std() / b.std():.8f}   "
          f"(both are float32 casts of float64 N(0,1)*scale)")
    return ok_all


def check_4b() -> bool:
    print()
    print("=== 4b: mc.layer_means vs whestbench.simulation.sample_layer_statistics ===")
    width, depth, n = 256, 32, 8192
    print(f"  width={width} depth={depth} n={n} "
          f"(whestbench chunk={_pick_chunk_size(width)}, mc default chunk=1024)")

    W = make_mlp(width, depth, 0)
    from whestbench.domain import MLP

    mlp = MLP(width=width, depth=depth, weights=[fnp.array(w) for w in W], seed=0)

    wb_means, wb_final, wb_var = sample_layer_statistics(
        mlp, n, fnp.random.default_rng(12345)
    )
    wb_means = np.asarray(wb_means, dtype=np.float64)
    wb_final = np.asarray(wb_final, dtype=np.float64)

    for chunk, label in ((1024, "mc default chunk=1024"), (4096, "chunk=4096")):
        ours, ourvar = layer_means(W, n, 12345, chunk=chunk, want_var=True,
                                   all_layers=True)
        ours32 = ours.astype(np.float32).astype(np.float64)
        d_final = ours32[-1] - wb_final
        d_all = ours32 - wb_means
        print(f"\n  [{label}]")
        print(f"    final layer  bit-identical after float32 cast : "
              f"{np.array_equal(ours32[-1], wb_final)}")
        print(f"    final layer  max|diff| = {np.abs(d_final).max():.3e}  "
              f"rms = {np.sqrt((d_final ** 2).mean()):.3e}  "
              f"(activation scale {wb_final.mean():.3f})")
        print(f"    all layers   max|diff| = {np.abs(d_all).max():.3e}")
        print(f"    per-layer rms diff (layers 1,2,8,16,31,32): " + " ".join(
            f"{np.sqrt((d_all[k] ** 2).mean()):.2e}" for k in (0, 1, 7, 15, 30, 31)))
        print(f"    avg_variance  ours={ourvar.mean():.10f}  "
              f"whestbench={wb_var:.10f}  diff={ourvar.mean() - wb_var:+.3e}")

    print()
    print("  NOTE the two known arithmetic differences in mc.layer_means:")
    print("   * mc.py:100  sums[li] += dst.sum(axis=0, dtype=dtype)  -- intermediate")
    print("     layers are summed in FLOAT32 per chunk; whestbench casts every layer")
    print("     to float64 before summing (simulation.py:129-130).  Final layer is")
    print("     float64 in both, so the SCORED number is unaffected.")
    print("   * mc rounds the mean only when the caller casts; whestbench always")
    print("     casts layer_means to float32 (simulation.py:137).")
    return True


def main() -> None:
    ok = check_4a()
    check_4b()
    print()
    print(f"VERDICT Claim 4a (same rng object): {'HOLDS' if ok else 'BROKEN'}")


if __name__ == "__main__":
    main()
