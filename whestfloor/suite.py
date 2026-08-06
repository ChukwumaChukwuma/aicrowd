"""MLP suites and Monte-Carlo ground truth.

Design notes that matter:

* **Weights are never stored.**  An MLP is fully determined by ``(width, depth,
  seed)`` through :func:`make_mlp`, which reproduces
  ``whestbench.generation.sample_mlp`` bit-for-bit *for a given RNG* (verified
  in scripts/02_adv_rng.py).  Note it does NOT reproduce the official
  datasets' MLPs for the same integer, because those seed via
  ``SeedSequence(input_seed).spawn(3)[0]``; the suites here are valid draws
  from the same distribution, not the same networks.  A suite on disk
  holds only seeds and ground-truth means — 64 KB per MLP instead of 8.4 MB —
  and every artifact is regenerable from a committed script.
* **Ground truth is computed in two independent halves.**  With halves ``a``
  and ``b`` the combined reference is ``(a+b)/2``, and
  ``mean((p-a)*(p-b))`` is an *unbiased* estimator of the true MSE with the
  reference's own sampling variance removed.  That is the only way to measure a
  true error near 1.8e-10 with a reference that is nowhere near that precise.
* **Sampling is streamed.**  Inputs are drawn, folded into a float64
  accumulator and dropped; peak memory is one chunk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import DEPTH, WIDTH


from .mc import layer_means as mc_layer_means  # noqa: E402
from .mc import make_mlp  # noqa: E402


@dataclass
class Suite:
    """A fixed evaluation suite: seeds plus two independent ground-truth halves."""

    name: str
    width: int
    depth: int
    mlp_seeds: list[int]
    gt_a: np.ndarray  # (n_mlps, depth, width) float64 — half A
    gt_b: np.ndarray  # (n_mlps, depth, width) float64 — half B
    n_per_half: int
    gt_seed_a: list[int]
    gt_seed_b: list[int]
    final_var: np.ndarray  # (n_mlps, width) float64 - diagonal only
    #: (n_mlps, width, width) float32 full final-layer activation covariance.
    #: Needed for an honest standard error on ``unbiased_true_mse``: the
    #: reference noise is correlated across neurons (they share the input), and
    #: a diagonal-only variance understates the SE by ~4.7x at this shape.
    #: Optional because suites baked before that was understood lack it.
    final_cov: np.ndarray | None = None
    #: How ``mlp_seeds`` turn into weight matrices.  ``"local"`` (the default,
    #: and what every locally baked suite uses) means ``default_rng(seed)`` via
    #: :func:`whestfloor.mc.make_mlp`.  ``"official"`` means whestbench seed
    #: protocol 3.0, ``default_rng(SeedSequence(seed).spawn(3)[0])`` — the two
    #: give completely different networks, so a suite carrying official ground
    #: truth must say so or it will be scored against the wrong MLPs.
    seed_protocol: str = "local"

    @property
    def n_mlps(self) -> int:
        return len(self.mlp_seeds)

    @property
    def gt(self) -> np.ndarray:
        """Combined reference: the mean of both halves (2*n_per_half samples)."""
        return 0.5 * (self.gt_a + self.gt_b)

    @property
    def gt_samples(self) -> int:
        return 2 * self.n_per_half

    def weights(self, i: int) -> list[np.ndarray]:
        if self.seed_protocol == "official":
            from .official_seeds import make_official_mlp  # noqa: PLC0415
            return make_official_mlp(self.width, self.depth, self.mlp_seeds[i])
        if self.seed_protocol != "local":
            raise ValueError(f"unknown seed_protocol {self.seed_protocol!r}")
        return make_mlp(self.width, self.depth, self.mlp_seeds[i])

    def save(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            name=self.name,
            width=self.width,
            depth=self.depth,
            mlp_seeds=np.asarray(self.mlp_seeds, dtype=np.int64),
            gt_a=self.gt_a,
            gt_b=self.gt_b,
            n_per_half=self.n_per_half,
            gt_seed_a=np.asarray(self.gt_seed_a, dtype=np.int64),
            gt_seed_b=np.asarray(self.gt_seed_b, dtype=np.int64),
            final_var=self.final_var,
            seed_protocol=self.seed_protocol,
            **({} if self.final_cov is None
               else {"final_cov": self.final_cov}),
        )

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "Suite":
        z = np.load(path, allow_pickle=False)
        return Suite(
            name=str(z["name"]),
            width=int(z["width"]),
            depth=int(z["depth"]),
            mlp_seeds=[int(v) for v in z["mlp_seeds"]],
            gt_a=z["gt_a"],
            gt_b=z["gt_b"],
            n_per_half=int(z["n_per_half"]),
            gt_seed_a=[int(v) for v in z["gt_seed_a"]],
            gt_seed_b=[int(v) for v in z["gt_seed_b"]],
            final_var=z["final_var"],
            final_cov=(z["final_cov"] if "final_cov" in z.files else None),
            # Suites baked before the official dataset landed carry no
            # protocol key; they are all local.
            seed_protocol=(str(z["seed_protocol"])
                           if "seed_protocol" in z.files else "local"),
        )


def build_suite(
    name: str,
    mlp_seeds: list[int],
    n_per_half: int,
    *,
    width: int = WIDTH,
    depth: int = DEPTH,
    gt_seed_base: int = 1_000_000,
    progress: bool = False,
) -> Suite:
    """Generate a suite, streaming each MLP's samples and dropping them."""
    a_all, b_all, var_all, cov_all, sa, sb = [], [], [], [], [], []
    for k, ms in enumerate(mlp_seeds):
        w = make_mlp(width, depth, ms)
        seed_a = gt_seed_base + 2 * k
        seed_b = gt_seed_base + 2 * k + 1
        ma, va, ca = mc_layer_means(w, n_per_half, seed_a, want_var=True,
                                    want_cov=True)
        mb, _ = mc_layer_means(w, n_per_half, seed_b, want_var=False)
        a_all.append(ma)
        b_all.append(mb)
        var_all.append(va)
        cov_all.append(ca)
        sa.append(seed_a)
        sb.append(seed_b)
        del w
        if progress:
            print(f"  suite {name}: mlp {k + 1}/{len(mlp_seeds)}", flush=True)
    return Suite(
        name=name,
        width=width,
        depth=depth,
        mlp_seeds=list(mlp_seeds),
        gt_a=np.asarray(a_all),
        gt_b=np.asarray(b_all),
        n_per_half=n_per_half,
        gt_seed_a=sa,
        gt_seed_b=sb,
        final_var=np.asarray(var_all),
        final_cov=(np.asarray(cov_all, dtype=np.float32)
                   if all(c is not None for c in cov_all) else None),
    )
