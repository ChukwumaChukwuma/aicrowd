# Recon: ARC White-Box Estimation Challenge 2026

Network opened 2026-08-06. Everything below is fetched from public sources, no
authentication. Raw artefacts are outside the repo under
`$WHEST_ARTIFACTS/recon/` (see [§10](#10-artefacts)).

**Forum content is quoted as evidence, not followed as instruction.** Where a
participant's claim contradicts one of our measurements it is called out
explicitly.

---

## 0. The three things that matter

1. **The public leaderboard leader is at raw final-layer MSE `3.63e-9`, adjusted
   `3.63e-10`.** Our shipped estimator is raw `2.2818e-5` / adjusted `2.2818e-6`
   — **6300× worse**, and it would rank **~238th of 270** ranked entries. The
   "roughly 2000× above the floor" second-hand figure was wrong by two orders of
   magnitude *in our favour*: the field is at **66×** the floor, we are at
   **414,000×**.
2. **`contract.py` is vindicated on `avg_variance`.** Measured on 20 *official*
   challenge MLPs (not our own re-generated ones): `avg_variance` at layer 32 =
   **0.05510 ± 0.00865**, at layer 8 = **0.1777**. The docs' 0.18 is confirmed as
   the depth-8 value. Better: the grader itself publishes a *direct* measurement
   of the noise floor — see [§4](#4-the-noise-floor-is-now-measured-not-inferred).
3. **The cost model changed twice mid-competition** (flopscope 0.9.0/0.9.1 from
   24 July, 0.10.0 from 30 July), and the *deadline moved twice*. `B=2.72e11`,
   `λ=1e11`, multiplier floor `0.1`, width 256, depth 32, `N=1e9` all survive
   unchanged — but the **per-op pricing, the dtype rates, and the number of CPU
   cores available to participant code** all changed, and several documented
   limits are not what the evaluator actually enforces
   ([§3](#3-contract-diff)).

---

## 1. Leaderboard reality

Source: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards`
(server-rendered React props, `leaderboards[id=1479]`, round 1428 "Phase 1",
`totalCount = 286`). Fetched 2026-08-06 ~18:5x UTC. Page 2 pulled via the site's
own GraphQL endpoint (`POST /graphql`, `entries(first:200, after:"MjAw")`).

`score` = `adjusted_final_layer_score`; `scoreSecondary` = raw `final_layer_mse`;
`mult` = score/scoreSecondary = mean `max(0.1, C/B)`.

| rank | who | adjusted | raw MSE | mult | subs | sub id |
|---|---|---|---|---|---|---|
| 1 | dpskv5 | **4.0e-10** | **3.6e-9** | 0.111 | 285 | 324969 |
| 2 | joe_wanza | 3.31e-9 | 4.37e-9 | 0.750 | 982 | 324936 |
| 3 | huang_chung_yi | 2.14e-8 | 2.14e-7 | 0.100 | 453 | 325039 |
| 4 | ednacob | 4.72e-8 | 9.31e-8 | 0.507 | 114 | 325145 |
| 5 | dstepanov | 5.90e-8 | 1.05e-7 | 0.562 | 411 | 324810 |
| 6 | ely2sh | 6.26e-8 | 5.57e-8 | 1.124 | 248 | 323861 |
| 7 | thylinao | 8.30e-8 | 1.72e-7 | 0.483 | 86 | 325077 |
| 8 | SKIBIDI_TOILET | 8.37e-8 | 1.99e-7 | 0.422 | 372 | 324895 |
| 9 | abhinav_gorrepati | 8.59e-8 | 1.93e-7 | 0.446 | 252 | 324107 |
| 10 | jtel | 9.12e-8 | 2.05e-7 | 0.446 | 221 | 324322 |
| … | | | | | | |
| 41 | SOX (natasha_stewart) | 1.60e-7 | 2.18e-7 | 0.734 | 162 | 319341 |
| 118 | jamesrahenry | 3.07e-7 | 2.22e-6 | 0.138 | 42 | 323492 |
| 182 | radiant-allomancer | 4.99e-7 | 4.43e-6 | 0.112 | 11 | — |
| **~238** | **(us, if we submitted)** | **2.2818e-6** | **2.2818e-5** | 0.100 | — | — |
| 251 | pscamillo | 2.45e-6 | 2.45e-5 | 0.100 | 1 | 314331 |
| 262 | **BASELINE: covariance propagation** | 6.622e-6 | 6.622e-5 | 0.100 | — | — |
| 273 | **BASELINE: mean propagation** | 9.635e-5 | 9.635e-4 | 0.100 | — | — |
| 277 | **BASELINE: random predictions** | 5.657e-2 | 0.5657 | 0.100 | — | — |

Our shipped `baseline_cov_prop_gain = 6.7456e-5` matches the official
covariance-propagation baseline `6.622e-5` to 2%, so our local channel is
calibrated correctly. **We are 2.9× better than the covariance-propagation
baseline and 6300× worse than the leader.** The whole analytic programme in
`state_of_play.md` has bought us about 30 leaderboard places out of 286.

### 1.1 The board moved 100× in the last 48 hours

`scoreHistory` for the top three (adjusted):

```
dpskv5           2026-08-05 09:44  5.08e-8 → 08-06 09:47  5.8e-9 → 08-06 16:26  4.0e-10
joe_wanza        2026-08-04 21:57  4.87e-8 → 08-05 14:06  6.6e-9 → 08-06 16:54  3.3e-9
huang_chung_yi   2026-08-06 03:04  1.14e-7 → 08-06 09:31  8.5e-8 → 08-06 16:02  2.14e-8
```

Three independent teams found ~10–100× on 5–6 August, after the v0.10.0 repricing.
Whatever this is, it is recent and it is not (mainly) the residual arbitrage —
see next.

### 1.2 Instrumented share: which top entries are real

Every submission page embeds the full per-MLP telemetry as JSON. Computing
`instrumented share = Σ flops_used / Σ effective_compute` (the diagnostic jtel
published in topic 18099) over all 100 MLPs:

| sub | who | adj | raw | C/B | **instr. share** | mean F | mean residual s | F spread | mean wall s |
|---|---|---|---|---|---|---|---|---|---|
| 324969 | dpskv5 #1 | 3.63e-10 | 3.63e-9 | 0.096 | **0.940** | 2.43e10 | 0.016 | 1.18 | 45.3 |
| 324936 | joe_wanza #2 | 3.31e-9 | 4.36e-9 | 0.754 | 0.439 | 9.01e10 | 1.149 | 1.000 | 55.5 |
| 325039 | huang_chung_yi #3 | 2.14e-8 | 2.14e-7 | 0.016 | 0.187 | 8.47e8 | 0.037 | 22.0 | 1.4 |
| 325145 | ednacob #4 | 4.72e-8 | 9.31e-8 | 0.507 | 0.929 | 1.28e11 | 0.098 | 1.10 | 44.2 |
| 324810 | dstepanov #5 | 5.90e-8 | 1.05e-7 | 0.561 | **8.4e-5** | 1.28e7 | 1.525 | 1.000 | 1.6 |
| 323861 | ely2sh #6 | 6.26e-8 | 5.57e-8 | 1.123 | **4.8e-5** | 1.47e7 | 3.054 | 1.000 | 3.1 |
| 325077 | thylinao #7 | 8.30e-8 | 1.72e-7 | 0.483 | 0.963 | 1.26e11 | 0.049 | 1.20 | 22.7 |
| 319341 | SOX #41 | 1.60e-7 | 2.18e-7 | 0.738 | 0.907 | 1.82e11 | 0.186 | 1.28 | 23.8 |

Reading:

- **#1 dpskv5 is genuine.** 94% instrumented, sits exactly at the 0.1 multiplier
  floor (`mean_score_multiplier` 0.10008, `mean_compute_utilization` 0.09587),
  burns 45 s of the wall on flopscope backend time. Its `per_layer_mse` is
  `[6.77e-10, 0.50, 0.62, …, 0.69, 3.63e-9]` — **it predicts only layer 1 (exact
  closed form) and layer 32, and leaves the middle 30 layers unfilled.** Only the
  final layer is scored, so that is rational.
- **#5 dstepanov and #6 ely2sh are residual arbitrage.** Instrumented share
  ~5e-5, `flops_used` bit-identical across all 100 MLPs, ~1.5–3 s of residual
  wall time doing all the real work at λ = 1e11 FLOP/s. These are the entries
  threads 18099 / 18108 / 18122 / 18132 are about.
- **#2 joe_wanza is half arbitrage.** 1.15 s residual = 1.15e11 charged FLOPs,
  56% of its cost, at a rate thylinao measures as ~2× under real single-core
  float32 GEMM throughput.
- **#3 huang_chung_yi shows the public/private overfitting signature.** Public-50
  `flops_used` CV = **1.43**, private-50 CV = **0.061**, public mean 1.53× the
  private mean. That is the pattern SKIBIDI_TOILET flagged in topic 18118 (public
  MLPs treated differently from private ones). Treat its 2.14e-7 as suspect.
- dpskv5, ednacob, thylinao, SOX all have public/private F ratios of 0.98–1.00
  with matched CV — clean.

**Practical consequence:** the honest, fully-instrumented frontier is roughly
`raw 3.6e-9` (dpskv5) → `9.3e-8` (ednacob) → `1.7–2.2e-7` (thylinao, SKIBIDI,
SOX). Even the *conservative* reading of the board puts the honest frontier
100–6000× ahead of our ship.

### 1.3 The matched-sampling reference, from the grader itself

Every submission's `aggregates.public` carries a constant field the docs never
mention:

```
"sampling_mse": 6.469470189211361e-07
```

It is identical across every submission (same 100-MLP suite). `whestbench`
documents `avg_variance` as *"the mean per-neuron variance at the final layer,
used as a normalisation baseline for `sampling_mse`"*
(`whestbench/src/whestbench/simulation.py:81-102`).

This is the number the whole board should be read against, because for a pure
sampler `mse × (C/B)` is **flat in N**: `mse = v/N`, `C = N·f`, so
`adjusted = v·f/B` independent of N. So `sampling_mse = 6.47e-7` **is** the
plain-Monte-Carlo adjusted plateau on this suite.

| | adjusted | × matched-sampling |
|---|---|---|
| dpskv5 #1 | 3.63e-10 | **0.0006×** |
| joe_wanza #2 | 3.31e-9 | 0.005× |
| huang_chung_yi #3 | 2.14e-8 | 0.033× |
| ednacob #4 | 4.72e-8 | 0.073× |
| SOX #41 | 1.60e-7 | 0.247× |
| jamesrahenry sampling entry | 3.07e-7 | 0.475× |
| **matched sampling (grader reference)** | **6.47e-7** | **1.00×** |
| **our ship** | **2.28e-6** | **3.53×** |
| covariance-propagation baseline | 6.62e-6 | 10.2× |

**Our shipped estimator is 3.5× WORSE than plain Monte Carlo at matched compute.**
That is the single most important strategic fact in this document and it is
absent from `state_of_play.md`, which never prices MC as a competitor.

> Cross-check against a participant claim: radiant-allomancer (topic 18085)
> writes *"Plain Monte-Carlo tops out at ≈2.8e-6 adjusted at any sample count,
> because above the floor the mse and the C/B multiplier cancel."* Their reasoning
> is right and matches the algebra above, but their **number is 4.3× worse than
> the grader's own `sampling_mse`** — presumably their MC implementation is
> dtype-naive (float64 bills 2×) or otherwise inefficient. **Use 6.47e-7, not
> 2.8e-6.**

---

## 2. Deadlines — and a live contradiction

| source | Phase 1 submission end |
|---|---|
| Official Rules timetable (`/challenge_rules`) | **July 31, 2026 23:59 UTC** (stale) |
| Site round metadata (`challenge.rounds[Phase 1].endDttm`) | **2026-08-07T23:59:00Z** |
| Forum announcement 18125, mohanty, 3 Aug (authoritative) | **10 August 2026, 23:59 UTC** |

Phase 2 `startDttm` on the site is still `2026-08-08T00:00:00Z`. bin_yong_bong
raised exactly this in topic 18130 and it has not been answered. **Today is
6 August.** If the submission gate keys off the site round metadata we have
~1.5 days, not ~4. Assume the earlier date.

Everything else, from 18125:

| item | date |
|---|---|
| Phase 1 submission deadline | 10 Aug 2026 23:59 UTC |
| Algorithmic-contribution write-up | 17 Aug 2026 23:59 UTC |
| Registration deadline | 5 Sep 2026 23:59 UTC |
| Team freeze | 5 Sep 2026 23:59 UTC |
| Phase 2 end | 19 Sep 2026 23:59 UTC |
| Private re-evaluation | 20–30 Sep 2026 |
| Results | ~1 Oct 2026 |

Rules §5.3: *"No scoring-affecting changes will be made in the final seven (7)
days of Phase 1 except security or integrity emergencies."* With a 10 Aug close
that window opened 3 August, i.e. the cost model should now be frozen.

---

## 3. Contract diff

Diffed against `whestfloor/contract.py`. **Green = contract.py is right and the
published docs are wrong.**

### 3.1 Confirmed unchanged (contract.py correct)

| constant | contract.py | confirmed by |
|---|---|---|
| `FLOP_BUDGET` 2.72e11 | ✓ | `evaluation.config_snapshot.flop_budget = 272000000000.0` on every submission; 18125 *"the following have not changed: the scoring formula, λ, the per-MLP FLOP budget, the 0.1 multiplier floor"* |
| `LAMBDA_FLOPS_PER_SECOND` 1e11 | ✓ | `whestbench/budget.py:LAMBDA_FLOPS_PER_SECOND = 1e11`; qi_zhang5 reproduced `C = F + 1e11·R` to max relative error **0.0** across 100 rows (topic 18129) |
| `MULTIPLIER_FLOOR` 0.1 | ✓ | `budget.py:score_multiplier`; 64 of 273 leaderboard rows sit at exactly mult 0.100 |
| `WIDTH` 256, `DEPTH` 32 | ✓ | `config_snapshot: {"n_mlps":100,"depth":32,"width":256}` |
| `GT_SAMPLES_OFFICIAL` 1e9 | ✓ | HF `metadata.json: "n_samples": 1000000000` |
| `score_multiplier` / `effective_compute` | ✓ byte-for-byte | `whestbench 0.14.0 src/whestbench/budget.py` |
| `MEMORY_LIMIT_MB` 65536 | ✓ | Rules §5.6 "64 GB RAM" |
| **`MEASURED_AVG_VARIANCE = 0.0551`** | ✓ | **independently reproduced on the official MLPs — [§4](#4-the-noise-floor-is-now-measured-not-inferred)** |

### 3.2 Discrepancies — docs wrong, we are right

| quantity | challenge docs say | truth | evidence |
|---|---|---|---|
| **multiplier floor** | **`s_m = MSE · max(0.5, C_m/B_m)`** — the Overview page states **0.5** | **0.1** | `budget.py`, and 64 of 273 board rows at mult exactly 0.100. **The Overview page is off by 5× on the single most important scoring constant.** |
| `avg_variance` | 0.18, floor "~2e-10" | 0.0551, floor 5.51e-11 | §4 |
| Example config | *"Hidden layers L: 8"*, *"B ≈ 3.4×10¹⁰"* | 32 layers, 2.72e11 | The Overview page was **never updated from the warm-up**. This is the direct provenance of the 0.18. |
| prize pool | "$100,000" | **$150,000+** | 18026: Phase 1 added $50k |

### 3.3 Discrepancies — the *deployed evaluator* differs from the shipped library

These matter because our sizing assumptions come from the library.

| item | library / docs | deployed evaluator | source |
|---|---|---|---|
| **over-budget behaviour** | `C_m > B` ⇒ predictions zeroed, multiplier forced to 1.0 (`scoring.py:826-843`, `is_combined_budget_exhausted`); Overview: *"if they don't, that MLP's prediction is replaced with zeros"* | **branch never fires.** Sub 323861: 97/100 MLPs over budget, max `C/B = 1.2327`, `combined_budget_exhausted: 0`, multiplier applied = uncapped `C/B`, median MSE on over-budget MLPs 4.24e-8 (real predictions, not zeros) | qi_zhang5, topic 18129, with a no-auth reproduction; unanswered by organisers |
| **`PREDICT_TIMEOUT_S = 30.0`** | `whestbench/cli.py` | not enforced as documented. thylinao read 100 rows of telemetry from 4 zero-failure submissions: max wall **64.818 s** with `time_exhausted: false`, `error_code: null`; means 22.6–49.3 s | topic 18125 posts #4–5, reproduction included. Our own fetch: dpskv5 mean wall 45.3 s / max 49.4 s, ednacob 44.2/49.6, joe_wanza 55.5/59.5 — all far past 30 s, none flagged |
| **`WALL_TIME_LIMIT_S = 60.0`** | `ContestSpec` default; Rules §5.6 "60-second hard wall-clock cap per MLP" | at least one MLP completed at **64.818 s** | same |
| smoke-test timeout | — | 30 s (different from the eval limit) | topic 18102 (unanswered) |
| per-array memory cap | undocumented 100 MiB, later raised | **4 GiB per array, 10M live arrays** | mohanty, topic 18039 |
| daily submission cap | "50 entries per team per UTC day", fixed window | mohanty confirms fixed-window; the *error message* was misleading | topic 18117 |

**Action for us:** `contract.py`'s `PREDICT_TIMEOUT_S = 30.0` is a **5× self-imposed
handicap** if we ever size anything against it. thylinao says exactly this:
*"We sized several approaches against a 30 s fatal ceiling and dropped them. If
the real envelope is 65 s, that was a self-inflicted 5× handicap taken on the
strength of a published number."* dpskv5 spends 45 s of wall at the 0.1 FLOP floor.

### 3.4 Cost-model changes we have *not* accounted for

`flopscope 0.10.0` + `whestbench 0.14.0` are what the evaluator runs
(`runtime_environment` on every submission: `flopscope_version 0.10.0`,
`whestbench_version 0.14.0`, `numpy 2.2.6`, `python 3.10.20`).

Verified directly from `flopscope/v0.10.0 src/flopscope/data/default_weights.json`:

```
dtype_rates: float32 1.0, float64 2.0, float16 1.0, int64 2.0,
             complex64 1.0, complex128 2.0, float128 4.0
op weights : matmul 1.0, einsum 1.0, dot 1.0, add 1.0, multiply 1.0,
             sqrt 1.0, sum 1.0, maximum 1.0, copyto 1.0, reshape 1.0,
             concatenate 1.0, take 4.0, sort 4.0, exp 16.0, zeros 0.0
```

Charged cost = `int(flop_cost × dtype_rate × complex_factor × weight)`.

Four things this changes for us:

1. **float64 costs 2×.** If any array in the chain is float64 the whole downstream
   product is float64. jamesrahenry's #314695 went from multiplier 0.169 → 0.300
   (raw MSE byte-identical) purely from this. Our estimator must be audited for
   dtype; `whestfloor` uses NumPy defaults in places.
2. **Data movement is no longer free.** `copy`/`fill`/`concatenate`/`reshape` = 1
   per element written; `gather`/`sort`/`histogram` = 4. Precomputed-table +
   gather schemes are dead.
3. **`flopscope.stats.*` silently promotes float32 → float64** — all 18 of
   `{cauchy,expon,laplace,logistic,norm,uniform} × {ppf,cdf,pdf}`
   (nkosi_ndwandwe, topic 18127, unanswered). One `norm.ppf` call reprices the
   entire rest of the estimator at 2×. Relevant if we ever use `norm.ppf` for
   inverse-CDF sampling — which the RQMC route requires.
4. **Participant code now gets one physical core (2 vCPUs); the flopscope backend
   gets seven physical cores (14 vCPUs).** Rules §5.6 and the launch announcement
   both still say "16 vCPUs". *"The one-core change clearly bit, and it moved the
   top of the leaderboard by roughly 6×"* (thylinao, 18125 #4).

Also from 18125 #4, verified by hash: the "Before/Now" table in the v0.10.0
announcement is **measured against a pre-0.9 baseline**, not against 0.9.1.
`default_weights.json` is byte-identical between v0.9.1 and v0.10.0
(sha256 `9ff1647a…`). The real 0.10.0 behavioural change is the `einsum(out=)`
casting fix.

---

## 4. The noise floor is now *measured*, not inferred

Two independent confirmations, both against the **official** competition data.

### 4.1 `avg_variance` on the official MLPs

keenanpepper baked N=1e8 higher moments for all 1000 MLPs of the official `full`
split and published them (`keenanpepper/arc-whestbench-higher-moments-2026`,
topic 18052). Each `.npz` carries `mean` (32,256), `m2 = E[h²]` (32,256), and
`official_alm` (the official N=1e9 `all_layer_means`, for cross-check).

I range-fetched only the `mean.npy` / `m2.npy` / `official_alm.npy` members out
of the 67 MB zip archives for the first 20 MLPs
(`$WHEST_ARTIFACTS/recon/hf/fetch_var.py`) and computed
`avg_variance[ℓ] = mean_i(m2[ℓ,i] − mean[ℓ,i]²)`. `official_alm` agrees with
`mean` to ≤1.5e-4 at layer 32, confirming alignment with the official dataset.

```
depth  1 : 0.68085 ± 0.00085
depth  2 : 0.50579 ± 0.00556
depth  4 : 0.33054 ± 0.00648        (our 02_adv_variance.py: 0.3292)
depth  8 : 0.17771 ± 0.00547   <--  the docs' "0.18"
depth 16 : 0.09401 ± 0.00676        (ours: 0.0999)
depth 32 : 0.05510 ± 0.00865   <--  contract.py: 0.0551 ± 0.0023
```

`contract.py`'s `MEASURED_AVG_VARIANCE = 0.0551` reproduces **to three
significant figures on the official instances**, and the depth-8 provenance of
0.18 is confirmed at 0.1777. That row of `state_of_play.md` is now not just
"measured here" but "measured on the graded distribution".

**One correction to `contract.py`:** `AVG_VARIANCE_CV = 0.48` is too low. On the
20 official MLPs the CV is **0.70** (sd 0.0387, range 0.0146–0.166, median
0.0437 — strongly right-skewed). Over a 100-MLP suite the floor is
`5.51e-11 ± 0.39e-11` (**7.0%**, not 4.8%). Over the 50 public MLPs, ±10%.

### 4.2 The grader publishes the floor directly, and it matches

`per_layer_mse[0]` for dpskv5 (sub 324969) and natasha_stewart (sub 319341) is
**identically `6.773e-10`**. Layer 1 is analytically exact —
`E[relu(z¹_i)] = σ_i/√(2π)` with `σ_i² = ‖W¹_{:,i}‖²` — so that number is *not an
estimator error at all*: it is **purely the ground-truth reference's own
Monte-Carlo variance**, `v₁/N_gt`.

```
measured on the grader's 50 public MLPs : 6.773e-10
predicted from our v₁ and N=1e9         : 0.68085 / 1e9 = 6.8085e-10
agreement                                : 0.5 %
```

That is an end-to-end, third-party, on-the-grader confirmation of the entire
`gt_noise_floor(v, N) = v/N` model in `contract.py`, including `N = 1e9`. Two
independent submissions agree to four significant figures.

**So the final-layer floor is `v₃₂(suite)/1e9`.** With `v₃₂ = 0.0551 ± 0.006`
for a 50-MLP suite: raw floor `5.5e-11 ± 0.6e-11`, adjusted `5.5e-12`.
The leader's raw `3.63e-9` is **66× the floor**; our `2.28e-5` is **414,000×**.

### 4.3 One open number: the sampling reference costs less than we price it

`sampling_mse = 6.4695e-7`. If the grader's reference sampler costs exactly
`contract.forward_pass_flops() = 4,198,400` per sample, then `N_B = 64,786` and
the implied suite `avg_variance` is `6.4695e-7 × 64,786 = 0.0419` — 1.5σ below
our 0.0551 ± 0.0087, possible but on the low side. The alternative reading is
that the reference sampler achieves `N ≈ 85,000` at budget, i.e. **our
`forward_pass_flops` over-prices Monte Carlo by ~31 %**.

Supporting the ~64.8k reading: the Phase 1 launch post says submissions must stay
*"~15,000× smaller than the sampling budget used for the reference targets"*, and
`1e9 / 64,786 = 15,435`. Either way **0.18 is excluded by a factor of 3.3–4.3**.

This is worth one cheap experiment: it directly prices every sampling-based
method, and sampling is now clearly the competitive route.

---

## 5. Mid-competition changes to the contract

Chronology from the forum, all organiser posts:

| date | change | topic |
|---|---|---|
| 2026-06-17 | **Warm-up → Phase 1: depth 8 → 32, budget 6.8e10 → 2.72e11**, dataset revision `v1-phase1`, evaluator flopscope 0.8.0rc1 / whestbench 0.12.0rc0, +$50k prizes | 18026 |
| 2026-06-23 | per-array memory cap raised 100 MiB → 4 GiB | 18039 |
| 2026-07-24 | **flopscope 0.9.0/0.9.1 live on evaluators**: dtype-aware billing (float64 = 2×), data movement priced (copy/fill/concat 1/elt, gather/sort 4/elt) | 18125 |
| 2026-07-30 | **flopscope 0.10.0 shipped**: symmetry-tag forgery voided, `einsum(out=)` pays for the write, `einsum(out=)` honours `casting=` (BREAKING) | 18125 |
| 2026-08-03 | **participant code restricted to 1 physical core / 2 vCPUs**; flopscope backend gets 7 physical cores. "moved the top of the leaderboard by roughly 6×" | 18125 |
| 2026-08-03 | deadline 31 Jul → 10 Aug; write-up → 17 Aug; registration + team freeze → 5 Sep | 18125 |
| 2026-08-03 | Phase 1 prize ranking moved to a **private re-evaluation on a fresh suite**, up to **2** nominations per team | 18125 |

**Explicitly unchanged** (18125, verbatim):

> "To avoid confusion, the following have not changed: the scoring formula, λ, the
> per-MLP FLOP budget, the 0.1 multiplier floor, and your permission to bundle
> your own libraries, native code, and precompiled artifacts."

**Explicitly reserved** (18125, verbatim):

> "We retain the right to adjust λ where necessary, particularly if participants
> intentionally try to exploit the available residual wall time."
>
> "During final evaluation, we may also make targeted changes to the cost model if
> we identify an exploit that creates a material gap between the charged cost and
> what the computation would cost in practice. Finding such an exploit will not, by
> itself, result in disqualification."

---

## 6. Nomination, private re-run, prizes

From 18125 and 18118 (mohanty), and Rules §5.4/§6:

- Every Phase 1 submission is graded on **50 public + 50 private MLPs** (100
  total). Only the public 50 appear on the live board.
- **Each team may nominate up to two submissions** for the Phase 1 private
  re-evaluation. If none is nominated, **the top two on the public board** are
  used. *(The Rules page still says "one (1)" — the forum supersedes it.)*
- The re-evaluation runs on a **freshly generated suite with private seeds never
  used in Phase 1 or 2**, *"same width and layer-count ranges and the same
  FLOP-budget calibration"*, size chosen by statistical power analysis.
- **"Prize rankings will be determined exclusively by the post-Phase 1 private
  re-evaluation results, not by any score displayed on the public leaderboard."**
- The nomination interface has **not yet appeared**; organisers said they will
  email each team. bin_yong_bong asked where it lives on 5 Aug (18130) —
  unanswered as of this recon. **Keep a note of the submission IDs.**
- Overfitting to public MLPs *"is allowed, but the resulting performance is
  unlikely to generalize"*.
- Ties: additional MLPs are generated until statistically separable, then shared
  rank with combined-and-split prizes.

### Instrumented share and eligibility (18132, mohanty, 5 Aug — verbatim)

> "A submission is not automatically ineligible merely because most of its charged
> cost comes from residual time. However, a submission deliberately built around
> unmetered numerical computation that exceeds what the residual-time charge is
> intended to represent may be adjusted or re-evaluated. In cases of deliberate
> circumvention, it may be disqualified."
>
> "The principle we plan to apply is that participant-side code should not use
> residual time as a second compute lane that effectively performs more than
> λ FLOP/s of numerical work while being charged only at λ."

Also (18122 #6): on the eval servers participants get **`flopscope-client`, not
`flopscope` core** — `fnp.ndarray` is a `RemoteArray` with slots
`('_handle_id','_shape','_dtype','_symmetry','__weakref__')` and **no `.base`**,
so the "raw NumPy array through flopscope.numpy" path is closed. You can still
bundle your own NumPy wheel, but on one core it is *"cheaper for you to use
flopscope"*.

### Algorithmic-contribution prize (18041, ARC guidance — verbatim)

> "We are most interested in 'mechanistic' estimation methods… We are **less
> interested in methods that rely heavily on sampling, fine-tuned constants,
> careful performance optimization, and opaque LLM-optimized code** (although
> clever sampling-based methods are of interest if they rely on interesting
> structural observations…)."
>
> "We will likely start by reading the technical writeup for the highest-scoring
> submissions, and award the prize to the submission where novel 'mechanistic'
> ideas made the largest improvement to performance over previously-known methods."

Mechanics: PDF + **exactly one graded submission ID**; email
`arc-whestbench@aicrowd.com` **or** post publicly on the forum (public posts are
additionally considered for the $500–5,000 Community Contribution prizes). Prize
requires releasing code under an OSI-approved licence; declining forfeits to the
next team. Townhall (18078): ARC *"will review roughly the top 10 submissions
(possibly more)"*.

Prize pool: Phase 1 $25k/$10k/$5k + $10k algorithmic; Phase 2 $50k/$20k/$10k +
$20k algorithmic; total $150,000+.

---

## 7. Mechanism catalogue from other participants

Ranked by how much it moves the score. Every entry names the mechanism, the
claimed number, and — where stated — **what the author says the mechanism does
NOT apply to**. All of this is participant testimony, not our measurement.

### Tier A — mechanisms that demonstrably work on the board

| # | mechanism | claimed effect | who / where | claimed NOT to apply to |
|---|---|---|---|---|
| A1 | **Rao-Blackwellise the final layer**: sample through L−1 layers, then close the last layer analytically with `E[relu(z)] = μΦ(α) + σφ(α)` instead of averaging `relu` samples | the dominant single win in every hybrid; radiant-allomancer's hybrid at matched budget reads 9.31e-6 raw where the *grader's* plain-MC plateau is 6.47e-6-equivalent | evaaaz 18053, radiant-allomancer 18085 §2.4 | — |
| A2 | **Exact layer 1**: `E[relu(z¹_i)] = σ_i/√(2π)`, `σ_i² = ‖W¹_{:,i}‖²`, zero sampling noise | free; verified on the grader (`per_layer_mse[0] = 6.77e-10 = the GT noise floor`) | evaaaz 18053, radiant-allomancer §2.2, dpskv5 & SOX telemetry | — |
| A3 | **RQMC input lattice** (randomly-shifted rank-1 Kronecker/Roberts lattice, Cranley-Patterson rotation, inverse-CDF to Gaussian; unbiased for every N because `frac(k·g+U) ~ U[0,1)ⁿ` exactly) | evaaaz: adjusted **4.10e-7** at C/B 0.42, 0/100 failures. radiant-allomancer: 1.40× over identical-budget iid MC (9.307e-6 → 6.653e-6) | evaaaz 18053, radiant-allomancer §2.1 | radiant-allomancer: **only 1.40×, not the 5–7× others report**, because covariance shrinkage already removes most of the variance RQMC targets — *"the two levers are partly redundant"* |
| A4 | **Neuron classification dead / on / kink** from moment propagation; treat "on" neurons linearly (no sampling) in the last two layers, sample only "kink" neurons; exploit the resulting sparsity by sorting columns by firing rate and grouping rows by active-column count for cheaper matmuls | adjusted **1.551e-7**, raw **2.18e-7** | natasha_stewart / SOX, 18106, sub 319341 (PDF attached) | *"our method does not easily transfer with architectural changes to the network"* (ablations over activation function and architecture in the PDF) |
| A5 | **Sample-measured Edgeworth on the scored row** — measure κ₃, κ₄ from the layer-L pre-activation samples you already have and apply `δ_skew = −κ₃μ/(6σ³)·φ(α)`, `δ_kurt = (κ₄/24)((α²−1)/σ³)φ(α)` | skew term removes 77.7% of the final non-Gaussian gap; skew+kurt 98.0% (validated against 2e6-sample brute force); worth −8% end to end | radiant-allomancer §2.5 | **the *transported* κ₃ is nearly useless** — see B2 |
| A6 | **Offline-trained per-neuron ridge corrector** on predict-time features (α and powers to α⁴, empirical-vs-analytic mean/cov gaps, per-MLP "glayer" descriptor). 14 terms, trained on 200 local MLPs, N_truth 1e7, loaded at setup for 0 FLOPs | +18% → +22.7%, holdout retention 95% | radiant-allomancer §2.6 | must be fit on **noisy predict-time** features; fitting on converged (variance-free) features removed 60% of converged bias offline yet **regressed the deployed score to 9.08e-6** |
| A7 | **Antithetic Sobol'** | used in the #41 pipeline through layer 30 | natasha_stewart 18106 | radiant-allomancer refutes **antithetic RQMC** at probe stage: the z/−z pairing perturbs the empirical covariance feeding the analytic suffix, worse at both generating vectors |
| A8 | **Cast the hot path to float32** | multiplier 0.300 → ~0.15 for a dtype-naive estimator, i.e. **2× on the adjusted score for free** | jamesrahenry 18097 erratum, mohanty 18125 | keep the inverse-CDF in float64: *"Sobol uniforms can round to exactly 1.0 in float32 and NaN the tail branch"* |

### Tier B — mechanisms other people killed, with mechanisms

| # | mechanism | outcome | who | why it fails |
|---|---|---|---|---|
| B1 | **Control variates anchored to the input** (Gaussian-optimal affine surrogate chain) | ρ = **0.25** at depth 32 vs a 0.95 gate; decays 0.86/0.55/0.41/0.25 at layers 1/4/8/32 | radiant-allomancer §4.1 | ReLU mixing decorrelates any input-anchored surrogate polynomially in depth. *"At depth 32 there is nothing left to couple to."* Killed an entire class before any build. Corroborated by evaaaz: linearisation/layer-1 CVs give ×1.0–1.2 |
| B2 | **Diagonal third-cumulant transport** `κ₃^pre = (W^⊙3)ᵀ κ₃^post` + Edgeworth | −1.44% (bar 5%) | radiant-allomancer §4.2 | the transport is lossy: diagonal-only propagation **loses ~98% of the skew** (true final-layer standardised g₁ rms 0.455, max 0.87). **This directly corroborates our ledger row `abec9ab5a2da`** (skewness is inherited, not generated) from a different direction |
| B3 | **Low-rank / sliced / CP third-cumulant menu** (the organisers' own suggestion) for the final mean | dead **by construction** | radiant-allomancer §4.2(a) | `E[relu(z^L_i)]` depends only on the *marginal* law of `z^L_i`, so cross-cumulants `κ₃[i,i,j]` cannot touch it. Beyond-diagonal structure only matters for deeper propagation. **This is a proof, and it applies to our κ₃ star-diagram term too** |
| B4 | Even an **exact** diagonal κ₃ Edgeworth term at the final layer | moves the scored mean by **1.59%** of the residual; propagation error in (μ,σ) is **99%** of the analytic mean residual, final-layer non-Gaussianity only **2%** | radiant-allomancer §4.2 | the ReLU mean is a near-linear functional of the low-order moments |
| B5 | **Layerwise mean re-anchoring** toward the analytic covprop mean after every ReLU | **4–9× worse**, monotone in anchor strength | radiant-allomancer §4.3 | the covprop mean is biased; injecting a biased anchor every layer compounds through all remaining layers. *"An unbiased anchor would rescue it; none exists."* **Independently reproduced by jamesrahenry** |
| B6 | **Chaining exact/step-exact Edgeworth corrections through a deep moment chain** | makes the chained estimator **worse** | jamesrahenry 18097 | the chain is an *error-compensating dynamical system*: per-layer errors anticorrelate and partially cancel, so it attenuates zero-mean noise but **amplifies small coherent biases at a measured ~16:1 exchange rate**. Corrections only work if fitted on the chain's own rolled-forward trajectories (DAgger-style) |
| B7 | **Subspace-split hybrid** exploiting the covariance rank collapse (participation ratio 128 → 5.2 over 32 layers, 25×) — the organisers' town-hall hypothesis | negative | jamesrahenry 18097 §6 | **sampling noise concentrates in the same low-rank subspace as the signal** (λᵢ/N, direction by direction), so the complement carries only ~1.7% of the sampling error and even a perfect analytic side is capped at ~2%. A low-rank method must either beat λᵢ/N in the leading directions or convert rank into FLOP reduction |
| B8 | **Analytic carrier of the true joint K3 structure** (validated against keenanpepper's published K3 tensors, reconstruction cosine 0.83) | still loses **4.5×** to a plain N=4096 MC probe, 10/10 IDs | jamesrahenry 18097 | the joint structure is real and compressible, but sampling it is cheaper than computing it |
| B9 | **Learned bias-corrector on deterministic block-covariance** (at the 0.1 floor) | cuts block-cov MSE only **2.3×**, needs ~17× to beat the sampler at the floor | evaaaz 18053 | the final-layer features cannot recover the whole-network higher cumulants that drive block-cov error |
| B10 | **Scalar multiplicative correction to covariance propagation** (~0.992) | −3× final-layer MSE at **zero added FLOPs**; adjusted **2.45e-6** | pscamillo 18063, sub 314331 | *"the exploitable predictive signal in the residual decays exponentially with depth (τ ≈ 5.1 layers) and is statistically indistinguishable from zero at the scored final layer."* **This is the same mechanism as our shipped `cov_prop_shrink(g=0.99975)` and it lands at 2.45e-6 — essentially identical to our 2.2818e-6.** Independent confirmation that this family tops out here |
| B11 | **Offline-learned quadrature weights** (unbiased via per-MLP Haar rotation) | max +1.7% held out (bar 8%); ridge-optimal weights collapse to uniform | radiant-allomancer §4.5 | *"the very rotation that makes the learned weights legal also Haar-scrambles the per-MLP directions, so no cross-MLP weight signal survives"* |
| B12 | **Eigen/PCA-ordered lattice** | 0.95–1.04× (bar 1.1×) | radiant-allomancer §4.1 | the Roberts lattice is dimension-balanced (αⱼ ranges only 0.996→0.369 across 256 dims), so there is no per-dimension quality gradient to exploit, unlike scrambled Sobol'; and 30 layers of mixing wash out any layer-1 alignment |
| B13 | **Median-of-means / robust moments** | 1.31e-5 and 1.23e-5, both worse than the plain mean | radiant-allomancer §4.4 | block-median inefficiency exceeds its outlier protection; the worst-MLP outliers are pure variance, not corruption |
| B14 | **Nonlinear corrector heads** (tanh-MLP, GBM stumps, RBF) | −2.95%, −2.0%, −1.31% vs −3.76% for a plain α³/α⁴ ridge | radiant-allomancer §4.5 | the reachable residual is a low-order polynomial in the pre-activation ratio α **living in its tail** (57% of neurons have \|α\|>3, range to ±6.5); clipping α to ±4 destroys the gain. Bounded/saturating models cannot represent it. *"The per-neuron detail the task hypothesized (weight-column norms, per-neuron skew/kurtosis shapes) is dead (≈0% signal beyond the polynomial)"* |
| B15 | **Exact Wick/Price off-diagonal ReLU cross-covariance** | −6.5% for the pure covariance-propagation estimator; **+0.36%** inside a sampling hybrid | radiant-allomancer §4.6 | the hybrid leans on the empirical MC covariance, so the accuracy of the analytic anchor is nearly irrelevant there. *(Note: they verified the leading-order gain method is exactly the k=1 truncation of the series — the same relation our Mehler `kmax` sweep found.)* |
| B16 | **Global/cross-neuron corrector features** (20 MLP-level spectral & weight-chain descriptors) | +0.16% (bar 5%) | radiant-allomancer §4.6 | saturated by a single per-MLP moment-gap descriptor; the residual is per-neuron α-tail, not global-spectral |
| B17 | **Plug-in curvature (delta-method) term** | RMS 4.6e-6 vs residual RMS 2.15e-3, correlation +0.019; ridge zeros its coefficient | radiant-allomancer §4.6 | RQMC already cut the sampling variance; the residual is closure bias, not plug-in curvature |
| B18 | **Korobov lattice**; **scrambled Sobol' vs Kronecker** | Korobov much worse (rank-1 quality is violently N-sensitive); Sobol' vs Kronecker per-row tied | evaaaz 18053 | — |
| B19 | **Fixed-shift derandomisation** | +4.43% (bar 5%) but swings −0.14% to +8.91% across shift seeds, **−311% unshifted** | radiant-allomancer §4.6 | real but shift-dependent; killed on the pre-committed shift |

### Where the two most careful campaigns say the remaining error lives

radiant-allomancer (§5): *"It is bias-dominated… ≥58%, and under the realistic
RQMC decay (p>1) 68–73%, into closure bias… What remains is the Gaussian-closure
bias of the analytic suffix itself… That error is not a function of any cheap
per-neuron statistic; it is a property of the propagation, and no corrector
applied at the final layer can reach it."*

jamesrahenry (18097): *"Sampling wins Phase 1's raw metric, and the write-up says
so plainly; the analytic track targets the regimes where sampling itself is
unreliable."* His trajectory-calibrated moment chain reaches **7.4e-6 held out,
~8.4× better than plain kprop at matched budget** — and still 3× worse than the
grader's matched-sampling line.

### The floor bet (arianvassili, 18105) — and why it is already broken

Filed 27 July, from ~#80: an unbiased budget-respecting estimator cannot beat a
per-pair variance of ~0.012, *"which works out to roughly 3.7e-7 adjusted"*, with
the structural core machine-checked in Lean 4 / Mathlib (zero `sorry`, zero
custom axioms) — but *"the floor's numerical size (the ~0.012) is measured, not
proven."*

That number is **consistent with our v = 0.0551**: at `N_pairs = 3,239` free
antithetic pairs, `0.012/3239 = 3.70e-6` raw / `3.70e-7` adjusted, and
`v(1+ρ)/2 = 0.012` gives ρ ≈ −0.56, which is what antithetic + CV should buy.
It is **not** consistent with 0.18.

The bet is already falsified on the public board: dpskv5 sits at **3.63e-10
adjusted, 1000× below the claimed floor**, at 94% instrumented share and at the
multiplier floor. Kerensa (18105 #2) had already said *"I really can't see how the
top 10 got there, it seems almost physically impossible. Either they are
overfitting or doing something I can't really imagine."* At the time yangxinyu_xie9
attributed it to unpriced FLOP arbitrage — **which was right in July and is no
longer the whole story in August**, given dpskv5's 0.94 instrumented share.

---

## 8. ARC's own posts

### `alignment.org/blog/competing-with-sampling/` (Nov 2025)

The **matching sampling principle**: for every architecture there exists a
*mechanistic* estimator `𝔾_M` such that, given a short explanation π
(`|π| ≤ O(|θ|)`), for every ε it runs in `O(Time(M_θ)/ε²)` and has
`E_c[(𝔾 − E_x[M(c,x)])²] ≤ ε² E_c[Var_x[M(c,x)]]`.

This is *exactly* the challenge's plateau: `mse × N_equiv ≤ v`, i.e.
`adjusted ≤ sampling_mse`. The board's `sampling_mse = 6.47e-7` is the MSP line.

Methodologically load-bearing, and things we have **not** tried:

1. **Cumulant propagation is ARC's own answer for random MLPs**, and they claim it
   works: *"We believe we have a mechanistic algorithm that competes with sampling
   for estimating the expected output of random MLPs on Gaussian inputs. (We have
   an empirical demonstration of competitiveness with sampling, and a proof sketch…)"*
   The order is set by *"d is the largest integer such that `n^d < 1/ε²`"*. At
   n=256 and `1/ε² ≈ 65,000`, `256² = 65,536 > 65,000` — **d = 1**. Their own rule
   says the affordable order at this shape is essentially first-order, which is a
   sharp caution against pushing κ₃/κ₄ diagrams at width 256.
2. **Hermite basis as the computational device**, not just an expansion. Their
   half-space solution only reaches `1/ε²` runtime *"by working in the Hermite
   basis of polynomials instead of the standard monomial basis"*. We used Hermite
   for the star-diagram factorisation; they use it for the *time complexity*.
3. **Deduction–projection**: successively model each layer by the best-fit model
   from a parameterised class. The "best-fit within a class" framing is broader
   than moment-matching and is what makes the estimator mechanistic.
4. **The ש-tensor / tensor-network route** (for the trained two-layer case, not
   ours): cumulants of a multivariate polynomial as a sum of all tensor
   contractions over networks of copies of `ש_d`; the open problems are computing
   the tensors and *"noticing when a tensor network can only contribute negligibly
   to the sum"*. Unsolved even at ARC.
5. **Polynomial approximation built one factor at a time** — the half-space
   algorithm approximates `H_{≤t} = H_{≤t-1}·h_t` by the best degree-d polynomial
   at each step, rather than expanding the whole product. A layerwise analogue for
   ReLU nets is not obviously what covariance propagation does.

Townhall additions (18078) — ARC's stated internal view:

- *"Cumulant propagation methods will probably beat quasi-Monte Carlo in the long
  run"*; low-rank refinement (spend compute on high-variance directions) is *"the
  most promising direction"*. **jamesrahenry tested and killed the natural
  implementation of exactly this (B7).**
- *"Pushing cumulant propagation to higher-order moments improves results
  initially but eventually diverges. ARC has explored Borel resummation as a
  possible fix but hasn't fully solved it."* **This is our ledger row
  `7c79215dca99` (Edgeworth-6 worse than Edgeworth-4) — ARC has independently hit
  the same wall.** They add it *"matters more for very narrow networks (e.g. width
  16) than for the current challenge parameters (width 256), where it shouldn't be
  a blocker"* — which our measurement contradicts: we see divergence at width 256.
- Offline MC pretraining on published moment datasets is **within the rules and
  untracked**, but *"seen as less likely to earn the algorithmic contribution
  prize"*.

### `alignment.org/blog/algzoo-…` (Jan 2026)

Different problem (8–1,408-parameter RNNs/transformers on algorithmic tasks).
Nothing directly transferable. The one relevant framing: ARC evaluates
explanations by *"mean squared error as a function of compute, which allows us to
compare the estimate with sampling"*, plus *surprise accounting* (a mechanistic
estimate with as few bits of total surprise as bits of optimisation used to select
the model). They note the exponential blow-up of linear regions with depth as the
reason brute-force region enumeration cannot match sampling at depth — the same
obstruction we hit.

Companion paper for the challenge: **arXiv:2605.05179**, Wu, Lecomte, Winer,
Robinson, Hilton, Christiano, *"Estimating the expected output of wide random MLPs
more efficiently than sampling"* (not fetched — arxiv.org was not on the reachable
list).

---

## 9. What this means for us

Stated plainly, because the recon overturns the framing in `state_of_play.md`:

1. **`state_of_play.md`'s "honest position" understates the gap by ~10⁴.** It says
   "5.9e5× the floor … not at the floor". The correct comparison is not to the
   floor, it is to **matched sampling at 6.47e-7 adjusted**, and we are **3.5×
   the wrong side of it**. Every mechanism in our ledger has been measured against
   the covariance-propagation baseline, which is 10× worse than plain MC.
2. **The entire top of the board is sampling or sampling-hybrid.** No purely
   analytic entry is near the frontier. The three cheap, universally-adopted
   levers we have never implemented — Rao-Blackwell final layer (A1), exact layer 1
   (A2), RQMC inputs (A3) — are worth roughly 10×, and are the price of entry.
3. **Two of our shipped/open mechanisms are independently refuted.**
   - Our shipped `cov_prop_shrink(g=0.99975)` is pscamillo's scalar correction
     (B10), and their independently-measured landing point (2.45e-6 adjusted) is
     within 8% of ours (2.28e-6). That family is at its ceiling.
   - Our κ₃ star-diagram term is targeted by B3, which is a *proof*: the scored
     final-layer mean depends only on the marginal law, so cross-cumulants cannot
     reach it. That is consistent with our ledger row `b43d07c159d6` finding the
     derived coefficient is never optimal.
4. **Our open line 1 (chaos-2 / generalized chi-square) is already closed by our
   own ledger** (`abf3fbff7d7a`, `b955a5c8eaea`) and independently by B7: 97% of
   the one-step error is "z is not Gaussian", not "the rectifier map is not
   Gaussian".
5. **Our open line 3 (offline-calibrated residual correction) is the one the field
   validates** (A6), but radiant-allomancer's warning is specific and costly: fit
   on **noisy predict-time features**, never on converged ones.
6. **Free wins available right now, independent of any algorithm change:**
   - audit every array for float64 (worth up to 2× on the multiplier);
   - stop sizing against `PREDICT_TIMEOUT_S = 30`; the real envelope is ≥60 s and
     the leader uses 45 s at the 0.1 FLOP floor;
   - `AVG_VARIANCE_CV` should be 0.70, not 0.48.

### Open measurements worth one experiment each

- **Price of an MC sample under flopscope 0.10.0.** `sampling_mse` implies either
  `N_B ≈ 85,000` (our `forward_pass_flops` over-prices MC by 31%) or a suite
  `avg_variance` of 0.0419. Both readings matter: MC pricing is now on the
  critical path. Note `reshape`/`copyto` now bill 1/element, which our
  `forward_pass_flops` does not include.
- **Does the over-budget cliff exist?** The deployed evaluator does not fire
  `combined_budget_exhausted` (qi_zhang5, 18129, unanswered). If it stays that
  way, `C > B` is a linear penalty, not a cliff — which changes the safe operating
  point. Do **not** build on this; it is an acknowledged bug that may be patched
  before the private re-run.
- **`v₃₂` on more than 20 MLPs.** The per-MLP distribution is right-skewed with
  CV 0.70; `fetch_var.py` will extend to any count (it crashed at MLP 18 on a
  transient fetch, restartable).

---

## 10. Artefacts

All under `$WHEST_ARTIFACTS/recon/` (=
`/tmp/claude-0/-home-user-aicrowd/373914b3-787b-5580-866d-92a576ded1af/scratchpad/_artifacts/recon`,
17 MB, outside the repo):

```
topics/t*.json, t*.txt     all 43 forum topics of category 2991, raw + readable
categories.json            discourse category index (challenge = id 2991)
cat2991_p*.json            category topic listings
aicrowd/leaderboards.html  + leaderboard_data.json (ranks 1-200)
aicrowd/lb_page2.json      ranks 201-286 via POST /graphql
aicrowd/main.{html,txt}    Overview tab (still the warm-up contract)
aicrowd/rules.txt          full Official Rules body (70 KB)
aicrowd/challenge_rules.html, insights.html, discussion.html, submissions.html
subs/<id>.{html,json}      full per-MLP telemetry for 11 submissions incl. top 8
arc/*.{html,txt}           the two ARC blog posts
flopscope/default_weights.json   v0.10.0 op weights + dtype rates
hf/metadata.json           official dataset metadata (n_samples 1e9, 256x32)
hf/kp_README.md            keenanpepper higher-moments dataset schema
hf/fetch_var.py            range-fetches mean.npy/m2.npy out of the 67 MB npz
hf/var40.log               per-MLP avg_variance, 20 official MLPs
hf/official_avg_variance.json
```

Key URLs:

- Forum category: `https://discourse.aicrowd.com/c/white-box-estimation-challenge-2026/2991`
- Cost-model + deadlines: `https://discourse.aicrowd.com/t/phase-1-update-flopscope-v0-10-0-cost-model-fixes-residual-time-safeguards-and-updated-deadlines/18125`
- Phase 1 launch: `https://discourse.aicrowd.com/t/phase-1-launch-deeper-models-and-increased-prizes/18026`
- Townhall summary: `https://discourse.aicrowd.com/t/townhall-summary-recording/18078` (video `https://youtu.be/pvCgbUJ-nTo`)
- Algorithmic prize: `https://discourse.aicrowd.com/t/algorithmic-contribution-prize-guidelines-how-arc-judges-these-prizes-discretion-technical-writeups-llm-usage/18041`
- Write-ups: `…/18053` (evaaaz), `…/18063` (pscamillo, PDF), `…/18085` (radiant-allomancer), `…/18097` (jamesrahenry, PDF + erratum), `…/18106` (natasha_stewart, PDF)
- Floor bet: `https://discourse.aicrowd.com/t/a-bet-on-a-floor-filed-before-the-reveal/18105`
- Accounting: `…/18099`, `…/18108`, `…/18122`, `…/18129`, `…/18132`
- Leaderboard: `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards`
- Higher moments: `https://huggingface.co/datasets/keenanpepper/arc-whestbench-higher-moments-2026`
