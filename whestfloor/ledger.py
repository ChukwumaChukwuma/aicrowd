"""Experiment ledger — append-only, schema-enforced, traceable to a commit.

Rules this module enforces mechanically, because they are the rules that make
a number mean something:

* An adjusted score may never be recorded without its raw MSE, its compute
  ratio and its seed.  ``append`` raises if any is missing.
* Every record names the script that produced it and the git commit the tree
  was at.  A number whose script is not committed is marked ``dirty`` and can
  be filtered out.
* Every experiment declares its ``acceptance_bar`` (a number) *before* the run.
  ``append`` refuses a record whose bar was filled in after the fact, and
  ``verdict`` is computed from the bar, never supplied by the caller.
* Records are immutable.  A re-run appends; it never edits.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

LEDGER_DIR = Path(__file__).resolve().parent.parent / "ledger"
LEDGER_PATH = LEDGER_DIR / "experiments.jsonl"

SCHEMA_VERSION = "1.0.0"

#: Fields without which a record is meaningless and is rejected outright.
REQUIRED = (
    "experiment",
    "script",
    "estimator",
    "acceptance_bar",
    "raw_final_layer_mse",
    "compute_ratio",
    "seed",
    "n_mlps",
    "gt_samples",
)


class LedgerError(ValueError):
    """Raised when a record violates the schema."""


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        return ""


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


def file_digest(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


@dataclass
class Experiment:
    """A pre-registered experiment.

    Construct this *before* the run, with the acceptance bar already filled in.
    The bar is a number and is never re-rolled: if a run misses it, the record
    says so.  ``Experiment.record`` is the only way to append.
    """

    name: str
    script: str
    hypothesis: str
    acceptance_bar: float
    bar_metric: str = "adjusted_final_layer_score"
    bar_direction: str = "lower_is_better"
    _t0: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.acceptance_bar, (int, float)):
            raise LedgerError("acceptance_bar must be a number fixed before the run")
        if self.bar_direction not in ("lower_is_better", "higher_is_better"):
            raise LedgerError(f"bad bar_direction {self.bar_direction!r}")
        self._t0 = time.time()

    def verdict(self, value: float) -> str:
        if self.bar_direction == "lower_is_better":
            return "PASS" if value <= self.acceptance_bar else "FAIL"
        return "PASS" if value >= self.acceptance_bar else "FAIL"

    def record(self, **fields: Any) -> dict[str, Any]:
        """Append one immutable result row and return it."""
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "id": uuid.uuid4().hex[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_s": round(time.time() - self._t0, 3),
            "experiment": self.name,
            "script": self.script,
            "script_sha256": file_digest(
                Path(__file__).resolve().parent.parent / self.script
            ),
            "hypothesis": self.hypothesis,
            "acceptance_bar": self.acceptance_bar,
            "bar_metric": self.bar_metric,
            "bar_direction": self.bar_direction,
            "git": git_state(),
        }
        payload.update(fields)
        return append(payload, _bar_owner=self)


def append(record: Mapping[str, Any], *, _bar_owner: Experiment | None = None) -> dict[str, Any]:
    """Validate and append one record.  Returns the stored dict."""
    if _bar_owner is None:
        raise LedgerError(
            "records must be appended through Experiment.record so the "
            "acceptance bar is pinned before the run"
        )
    rec = dict(record)
    missing = [k for k in REQUIRED if k not in rec or rec[k] is None]
    if missing:
        raise LedgerError(f"record missing required fields: {missing}")

    if "adjusted_final_layer_score" in rec:
        for k in ("raw_final_layer_mse", "compute_ratio", "seed"):
            if rec.get(k) is None:
                raise LedgerError(
                    "an adjusted score may not be recorded without "
                    f"{k}; this is the reporting rule the ledger exists to enforce"
                )

    bar_value = rec.get(rec["bar_metric"])
    if bar_value is None:
        raise LedgerError(f"bar_metric {rec['bar_metric']!r} absent from record")
    rec["verdict"] = _bar_owner.verdict(float(bar_value))

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=float) + "\n")
    return rec


def read_all() -> list[dict[str, Any]]:
    if not LEDGER_PATH.is_file():
        return []
    out = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
