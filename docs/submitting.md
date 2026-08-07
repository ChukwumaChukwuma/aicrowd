# Submitting to AIcrowd

Two routes, both producing the same artifact:

* **Route A — from this sandbox**, where `pypi.org` is 403 and the official
  `whest` CLI cannot be installed. `scripts/35_package.py` builds the tarball
  by hand, `scripts/36_submit.py` ships it over the REST API. Both are
  stdlib-only replicas of the whestbench code paths.
* **Route B — from your own machine**, with the real `whest` CLI. Three
  commands, nothing hand-rolled.

Both routes upload **one tarball**. There is no git push to a per-participant
GitLab repo and no separate metadata file — the challenge's own words are
"You upload a single tarball produced by `whest package`", and the starter kit
confirms the CLI path ships it straight over the API.

---

## The facts that govern shipping

Read off the live challenge page and verified against the live API on
2026-08-07.

| Thing | Value |
|---|---|
| Challenge slug | `arc-white-box-estimation-challenge-2026` |
| Numeric challenge id | `1174` (`GET https://api.aicrowd.com/challenges/?slug=…`) |
| Registration | confirmed `{"registered": true}` for participant `nnamdi` (id `514915`) |
| **Daily limit** | **50 entries per team, per UTC day, per phase** |
| Phase 1 close | **Aug 10 2026 · 23:59 UTC** (extended from Jul 31) |
| Phase 1 private re-run | Aug 11–17, on **up to 2 nominated submissions per team** |
| Phase 2 close | Sep 19 2026 · 23:59 UTC |
| Submission cap | 50 MiB total, 50 files (`whestbench/limits.py`) |
| Grader hardware | CPU-only, 1 core (2 vCPU) for your code, 7-core flopscope backend, 64 GB, **no network** |
| Hard wall cap | 60 s per MLP |
| Setup window | ~5 s, and it includes interpreter start and imports |
| Budget | `B = 2.72e11` FLOPs/MLP, `λ = 1e11` FLOPs/s |
| Score | `mean_m [ final_layer_mse_m × max(0.1, C_m/B) ]`, lower is better |

Two notes on the scoring floor. The Overview prose writes `max(0.5, C/B)`;
the page's own leaderboard component says the 10% floor "is the ACTUAL scoring
rule", and `whestbench/budget.py::score_multiplier` is `max(0.1, C/B)`. The
shipped estimator is tuned to the 0.1 floor, which is the one that is in the
code.

The 50/day limit means **ship early and often**. The public leaderboard leaders
have 285–1005 entries. Nothing is gained by hoarding attempts; the only
submissions that decide anything are the (up to) two nominated for the private
re-run.

---

## Route A — submitting from this sandbox

### A1. Environment

```bash
. /tmp/claude-0/-home-user-aicrowd/373914b3-787b-5580-866d-92a576ded1af/scratchpad/prefix/runenv.sh
export WHEST_ARTIFACTS=/tmp/claude-0/-home-user-aicrowd/373914b3-787b-5580-866d-92a576ded1af/scratchpad/_artifacts
cd /home/user/aicrowd
```

`$PY` is Python 3.12 with numpy and flopscope on `PYTHONPATH`. `whestbench`
itself is *not* installed and cannot be — that is why both scripts exist.

### A2. Package, validate, audit, sandbox-test

```bash
$PY scripts/35_package.py
```

One command, four stages, exit 0 only if every one passes:

1. **package** — folder-mode bundle of `submission/` to a flat `.tar.gz` with a
   `manifest.json` member, per-file sha256, `_BUILTIN_IGNORES` +
   `_SECRET_IGNORES` applied, 50 MiB / 50 file caps enforced, entrypoint class
   resolved by loading `estimator.py` and binding `predict(mlp, budget)`.
2. **validate** — a re-implementation of `whestbench/validation.py::validate_package`:
   re-opens the archive, re-hashes every member against the manifest, checks
   the entrypoint module file exists and that no `files[]` entry is a directory.
   This is the same gate `whest submit` runs before it uploads.
3. **audit** — prints the exact member list with sizes and hashes, then refuses
   any non-regular member, unsafe path, `__pycache__` / `_artifacts` / `.git`
   fragment, credential-shaped filename, or credential-shaped *content*
   (PEM blocks, AWS keys, `Authorization: Token …` literals, and so on). Also
   AST-scans every shipped `.py` for imports on **every** code path, including
   lazy ones inside functions.
4. **sandbox** — the real proof; see below.

Useful flags:

```bash
$PY scripts/35_package.py --out /tmp/mysub.tar.gz     # explicit path, no -latest copy
$PY scripts/35_package.py --skip-sandbox              # packaging only, fast
$PY scripts/35_package.py --keep-extract              # leave the extraction dir to poke at
$PY scripts/35_package.py --verify-only $WHEST_ARTIFACTS/submission-latest.tar.gz
```

Output lands in `$WHEST_ARTIFACTS`:

* `submission-<UTC timestamp>.tar.gz` — the immutable build
* `submission-latest.tar.gz` — a byte copy at a stable path, for the submitter
* `package_report.json` — machine-readable record of every stage

Nothing is written into the repo. `*.tar.gz` and `_artifacts/` are gitignored
anyway, but the artifacts directory is the belt to that suspenders.

### A3. Dry-run the submission

Dry run is the **default**; `--send` is required to transmit.

```bash
$PY scripts/36_submit.py --tarball $WHEST_ARTIFACTS/submission-latest.tar.gz
```

This performs identity → challenge resolution → registration check → presigned
upload request, i.e. it proves the credential path and the tarball are both
good, and then stops without uploading or creating anything.

### A4. Send it

```bash
$PY scripts/36_submit.py \
    --tarball $WHEST_ARTIFACTS/submission-latest.tar.gz \
    --send --watch \
    -m "sparse MC + layer-1 Hermite CVs + offline head"
```

`--watch` polls until the submission reaches a terminal grading state.
To poll an existing one later:

```bash
$PY scripts/36_submit.py --status <submission_id> --watch
```

### A5. Credential handling

`scripts/36_submit.py` reads the key from `AICROWD_API_KEY`, or failing that
from a file outside the working tree (default `~/.aicrowd_key`, `chmod 600`).
It **refuses** a key file located inside the repository. The key is never
printed, never written anywhere, never placed on a command line, and is
scrubbed out of every error message before it is emitted.

Do not `export` the key in a shell whose history is persisted, and never pass
it as a CLI argument — an argument is visible in `ps` to every process on the
box. Put it in a file outside the tree, or feed it in as:

```bash
read -rs AICROWD_API_KEY && export AICROWD_API_KEY
```

Get the key from <https://www.aicrowd.com/participants/me/customize>.

---

## Route B — submitting from your own machine with the `whest` CLI

This is the supported path and needs no code from this repo.

```bash
git clone https://github.com/AIcrowd/whest-starterkit.git
cd whest-starterkit
uv sync
```

Copy the two shipped files from this repo into the starter kit (or point
`--estimator` straight at this repo's `submission/` folder — folder mode ships
the whole directory, which is what we want because the estimator loads
`corrector.npz` at runtime):

```bash
uv run whest validate --estimator /path/to/aicrowd/submission/estimator.py
uv run whest package  --estimator /path/to/aicrowd/submission \
                      --output submission.tar.gz
uv run whest login                       # paste the API key once; stored in
                                         # ~/.config/aicrowd-cli/config.toml
uv run whest submit   --estimator /path/to/aicrowd/submission --watch
```

(`whest validate` wants the *file*: it sets `ctx.submission_dir` to that file's
parent, which is how `corrector.npz` gets found. `package` and `submit` want
the *folder*, so the whole directory ships. The flag is `--output`, not `-o`.)

`whest submit --estimator <folder>` packages and uploads in one step. To ship a
tarball this repo already built:

```bash
uv run whest submit submission.tar.gz --watch
```

Two things worth doing first, if you have the bandwidth — they are closer to
the grader than anything we can run here:

```bash
uv run whest run --estimator /path/to/aicrowd/submission \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
    --split mini --runner subprocess
uv run whest run --estimator /path/to/aicrowd/submission ... --runner docker
```

The browser also works: <https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/new>
accepts the same `submission.tar.gz`.

---

## The wire protocol, if you ever need raw curl

Verified against the whestbench source (`aicrowd_client.py`) and against the
live server. Auth is `Authorization: Token <key>` — **not** `Bearer`.
`{rails}` = `https://www.aicrowd.com/api/v1`.

```bash
SLUG=arc-white-box-estimation-challenge-2026
RAILS=https://www.aicrowd.com/api/v1
# AICROWD_API_KEY must already be exported. The line below references the
# variable; it never expands the key into anything that is logged.

# 0. identity  ->  {"id": <participant_id>, ...}
curl -sS -H "Authorization: Token $AICROWD_API_KEY" "$RAILS/api_user"

# 0b. registration (numeric challenge id 1174)
curl -sS -H "Authorization: Token $AICROWD_API_KEY" \
  "https://api.aicrowd.com/challenges/1174/participant?participant_id=<pid>"

# 1. presign  ->  {"data": {"url": ..., "fields": {...}}, "success": true}
curl -sS -H "Authorization: Token $AICROWD_API_KEY" \
  "$RAILS/submissions?challenge_id=$SLUG"

# 2. upload to S3 — multipart, every presigned field plus the file, and NO
#    Authorization header (different host). Substitute ${filename} in
#    fields["key"] locally so you know the final object key.
curl -sS -X POST "<data.url>" \
  -F key="<data.fields.key with \${filename} replaced>" \
  -F <every other presigned field> \
  -F file=@submission.tar.gz

# 3. create  ->  {"data": {"submission_id": <id>, ...}, "success": true}
curl -sS -X POST -H "Authorization: Token $AICROWD_API_KEY" \
  -H 'Content-Type: application/json' "$RAILS/submissions" \
  -d '{"challenge_id":"'"$SLUG"'",
       "submission":{"description":"..."},
       "submission_files":[{"submission_file_s3_key":"<key>"}]}'

# 4. poll
curl -sS -H "Authorization: Token $AICROWD_API_KEY" "$RAILS/submissions/<id>"
```

Note `challenge_id` in steps 1 and 3 is the **slug**, not the numeric id; the
numeric id is only used by the `api.aicrowd.com` registration check.

Step 4 returns `grading_status_cd` ∈ `initiated | submitted | ready | graded |
failed`, plus `score` (the adjusted final-layer score), `score_secondary`
(all-layer MSE) and `grading_message`. Poll until `graded` or `failed`.

All four endpoints were probed live from this sandbox without a key and each
returned a clean `401` with the documented error body, which confirms both
reachability and the auth scheme:

```
GET  www.aicrowd.com/api/v1/api_user                 -> 401 "HTTP Token: Access denied."
GET  api.aicrowd.com/challenges/?slug=…              -> 401 {"message": "Bad token"}
GET  www.aicrowd.com/api/v1/submissions?challenge_id -> 401 {"error": "Authentication required …"}
GET  www.aicrowd.com/api/v1/submissions/1            -> 401 "HTTP Token: Access denied."
```

---

## What the sandbox stage actually proves

The grader runs the estimator in a subprocess worker whose environment
provides **only** flopscope, the whestbench API, and a reduced stdlib. No
numpy, no scipy, no torch, no this repository. A submission that leans on any
of those passes every local test — because locally they are all importable —
and then dies on the grader, one zeroed MLP at a time.

Stage 4 of `scripts/35_package.py` reproduces that environment:

* The tarball is **extracted to a temp dir** and everything is loaded from
  there, not from `submission/`. What is tested is the artifact, not the
  working tree.
* A **fresh interpreter** is spawned with `-P` (no cwd, no script dir on
  `sys.path`) and a `PYTHONPATH` from which every entry inside the repo has
  been stripped. `-I`/`-E` are deliberately *not* used, because flopscope
  itself lives on `PYTHONPATH` here and would vanish with them.
* A **positive control** runs first: the child asserts
  `importlib.util.find_spec("whestfloor") is None`. If the repo were still
  reachable the probe aborts rather than reporting a false pass.
* An **import firewall** is installed into the estimator module's own
  `__builtins__`, so every `import` statement executed by submission code
  routes through it — including lazy imports inside functions and imports of
  modules already in `sys.modules`. numpy, scipy, torch, pandas, sklearn, jax,
  whestfloor and friends raise `ImportError("SANDBOX: …")`. flopscope's own
  internals are untouched, because they carry their own builtins.
* The firewall itself has a **positive control**: a deliberate `import numpy`
  must be blocked before the estimator is loaded, or the probe aborts.
* The class is then resolved the way `whestbench/loader.py` does it (submission
  dir on `sys.path`, `spec_from_file_location`, prefer a class named
  `Estimator`), `setup()` is called with a real `SetupContext`, and `predict`
  is run on a freshly generated He-Gaussian 32×256 MLP inside a
  `flopscope.BudgetContext` at `B = 2.72e11` with the 60 s wall cap.

`whestbench` is not pip-installable here, so the probe supplies a minimal
stand-in exposing exactly `BaseEstimator` and `SetupContext`. The grader
supplies the real package; the estimator only touches those two names.

The setup clock is stamped the way `SubprocessRunner.start` stamps it: from
immediately before `Popen` to the moment `setup()` answers. That window
therefore covers interpreter start, the flopscope import, loading
`estimator.py` and `setup()` itself — which is exactly what the 5 s cap
covers. It is a **lower bound**, because the real worker also imports the
whestbench package inside that same window and this box cannot install it.

### Negative controls

The test was checked against two deliberately poisoned copies of the
submission folder, to prove it can actually fail:

| Poison | Caught by |
|---|---|
| `import numpy as _np` **inside `predict()`** (never executed at import time) | static AST scan *and* the runtime firewall, during predict |
| `import whestfloor.kernels` at module level | static AST scan *and* the runtime firewall, during load |

Both produced `NOT SHIPPABLE` and exit code 1. The clean submission exits 0.

---

## Pre-flight checklist

```bash
# 1. the research kernel and the shipped estimator are still the same algorithm.
#    pytest is not installable here (pypi 403), so run the test functions directly.
$PY - <<'PY'
import importlib.util, pathlib, traceback
spec = importlib.util.spec_from_file_location(
    "t", pathlib.Path("tests/test_submission_parity.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fail = 0
for n in sorted(x for x in dir(m) if x.startswith("test_")):
    try:
        getattr(m, n)(); print("PASS", n)
    except Exception:
        fail += 1; print("FAIL", n); traceback.print_exc()
raise SystemExit(fail)
PY

# 2. package + validate + audit + sandbox
$PY scripts/35_package.py

# 3. dry run the wire protocol
$PY scripts/36_submit.py --tarball $WHEST_ARTIFACTS/submission-latest.tar.gz

# 4. ship
$PY scripts/36_submit.py --tarball $WHEST_ARTIFACTS/submission-latest.tar.gz \
    --send --watch -m "<what changed>"
```

Then confirm on the leaderboard, and remember that the score that matters is
the private re-run of the nominated submissions, not the public board.

## Known cosmetic deviation

`build_manifest` fills `whestbench_version` and `flopscope_version` from
installed *distribution metadata*. Neither package is pip-installed in this
sandbox, so both fields read `"unknown"` — which is precisely what the real
`build_manifest` would write if it ran here. `validate_package` does not
inspect either field, and neither does the grader's integrity check; they are
provenance metadata only. Route B produces real version strings.
