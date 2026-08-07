#!/usr/bin/env python3
"""Build, verify and sandbox-test the AIcrowd submission tarball.

`pypi.org` is 403 in this sandbox, so the official ``whest`` CLI cannot be
installed and ``whest package`` cannot be run.  This script is a faithful,
hand-built replica of the packaging contract, read out of the whestbench
source rather than the docs:

    whestbench/packaging.py   resolve_submission, collect_submission_files,
                              enforce_submission_caps, build_manifest,
                              _BUILTIN_IGNORES, _SECRET_IGNORES, _finalize_package
    whestbench/validation.py  validate_package
    whestbench/limits.py      MAX_SUBMISSION_BYTES = 50 MiB, MAX_SUBMISSION_FILES = 50

It runs four stages and refuses to leave a tarball behind if any of them fail:

  1. PACKAGE   folder-mode bundle of ``submission/`` -> flat ``.tar.gz`` with a
               ``manifest.json`` member; same layout, same ignore rules, same
               manifest fields, per-file sha256.
  2. VALIDATE  a re-implementation of ``validation.py::validate_package``:
               re-open the archive, re-hash every member, check the entrypoint
               module is present and that no ``files[]`` entry is a directory.
  3. AUDIT     what is actually about to leave the machine: exact file list,
               byte counts, credential scan, ``_artifacts`` / ``__pycache__``
               check, license check.
  4. SANDBOX   extract to a temp dir and load ``estimator.py`` in a FRESH
               interpreter with the repo off ``sys.path`` and an import
               firewall that denies numpy/scipy/torch/whestfloor to the
               estimator's own module globals.  Times ``setup()`` against the
               5 s window (which also covers interpreter start and imports),
               runs ``predict`` on a real 32x256 MLP, and checks shape,
               finiteness and effective compute against the budget.

Usage
-----
    . <scratch>/prefix/runenv.sh
    export WHEST_ARTIFACTS=<scratch>/_artifacts
    $PY scripts/35_package.py

    $PY scripts/35_package.py --out /tmp/mysub.tar.gz --skip-sandbox
    $PY scripts/35_package.py --verify-only $WHEST_ARTIFACTS/submission-latest.tar.gz

Exit code is 0 only when every stage passes.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# whestbench/limits.py -- verbatim
# --------------------------------------------------------------------------

MAX_SUBMISSION_BYTES: int = 50 * 1024 * 1024
MAX_SUBMISSION_FILES: int = 50

# --------------------------------------------------------------------------
# whestbench/packaging.py -- ignore sets, verbatim
# --------------------------------------------------------------------------

_BUILTIN_IGNORES: Tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
    "*.tar.gz",
    "*.tgz",
    "*.zip",
    ".whestignore",
    ".gitignore",
    "manifest.json",
)

_SECRET_IGNORES: Tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".pypirc",
    ".npmrc",
    ".aws",
    ".ssh",
    ".gnupg",
    "credentials",
    "credentials.*",
)

# The estimator contract, from whestbench/cli.py + runner.py + the challenge page.
WIDTH = 256
DEPTH = 32
FLOP_BUDGET = 272_000_000_000
LAMBDA_FLOPS_PER_SECOND = 1e11
SETUP_TIMEOUT_S = 5.0  # ResourceLimits.setup_timeout_s in whestbench/cli.py
WALL_TIME_LIMIT_S = 60.0  # "Hard cap 60 s per MLP" (challenge Official facts)
API_VERSION = "2.0"

# --------------------------------------------------------------------------
# tiny reporting helpers
# --------------------------------------------------------------------------

_FAILURES: List[str] = []
_WARNINGS: List[str] = []


def head(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m" if sys.stdout.isatty() else f"\n{title}")
    print("-" * max(24, len(title)))


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def warn(msg: str) -> None:
    _WARNINGS.append(msg)
    print(f"  WARN  {msg}")


def bad(msg: str) -> None:
    _FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def info(msg: str) -> None:
    print(f"        {msg}")


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.2f} MiB"


# --------------------------------------------------------------------------
# STAGE 1 -- packaging (whestbench/packaging.py)
# --------------------------------------------------------------------------


def _installed_version(distribution: str) -> str:
    """packaging.py::_installed_version -- 'unknown' when not pip-installed."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _sha256(path: Path) -> str:
    """packaging.py::_sha256 -- refuses anything that is not a regular file."""
    if not path.is_file():
        raise ValueError(
            f"Cannot hash {path}: a submission manifest entry must be a regular file, "
            f"not a directory or special file."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_submission(estimator_path: "str | Path") -> Tuple[Path, Path, str]:
    """packaging.py::resolve_submission."""
    target = Path(estimator_path).resolve()
    if target.is_dir():
        entry = target / "estimator.py"
        if not entry.is_file():
            raise FileNotFoundError(
                f"Folder submission requires an estimator.py in {target}, but none was found."
            )
        return target, entry, "folder"
    if target.is_file():
        return target.parent, target, "file"
    raise FileNotFoundError(f"Estimator path not found: {target}")


def _load_ignore_patterns(root: Path) -> List[str]:
    """packaging.py::_load_ignore_patterns.

    NOTE the scope: only ``<root>/.gitignore`` and ``<root>/.whestignore`` are
    read -- the repository-root ``.gitignore`` is NOT consulted.  That is why
    the repo-level ``*.npz`` rule does not drop ``submission/corrector.npz``.
    """
    pats = list(_BUILTIN_IGNORES)
    for fname in (".gitignore", ".whestignore"):
        f = root / fname
        if f.is_file():
            pats += [
                ln.strip()
                for ln in f.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
    return pats


def _is_ignored(rel: Path, patterns: List[str]) -> bool:
    """packaging.py::_is_ignored."""
    parts = rel.parts
    name = rel.name
    for pat in patterns:
        p = pat.rstrip("/")
        if any(fnmatch.fnmatch(seg, p) for seg in parts):
            return True
        if fnmatch.fnmatch(name, p) or fnmatch.fnmatch(str(rel), p):
            return True
    return False


def _is_secret(rel: Path) -> bool:
    """packaging.py::_is_secret -- case-insensitive, per path segment."""
    parts = [seg.lower() for seg in rel.parts]
    name = rel.name.lower()
    for pat in _SECRET_IGNORES:
        p = pat.lower().rstrip("/")
        if any(fnmatch.fnmatch(seg, p) for seg in parts) or fnmatch.fnmatch(name, p):
            return True
    return False


def collect_submission_files(root: "str | Path") -> List[Path]:
    """packaging.py::collect_submission_files -- sorted absolute paths."""
    root = Path(root).resolve()
    patterns = _load_ignore_patterns(root)
    out: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_ignored(rel, patterns) or _is_secret(rel):
            continue
        out.append(path)
    return out


def find_secret_files(root: "str | Path") -> List[Path]:
    """packaging.py::find_secret_files -- what was dropped for security."""
    root = Path(root).resolve()
    out: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if _is_secret(path.relative_to(root)):
            out.append(path)
    return out


def enforce_submission_caps(files: List[Path]) -> None:
    """packaging.py::enforce_submission_caps."""
    n = len(files)
    if n > MAX_SUBMISSION_FILES:
        raise ValueError(
            f"Submission has {n} files, over the {MAX_SUBMISSION_FILES}-file cap."
        )
    total = sum(f.stat().st_size for f in files)
    if total > MAX_SUBMISSION_BYTES:
        raise ValueError(
            f"Submission is {total / 1e6:.1f} MB, over the "
            f"{MAX_SUBMISSION_BYTES / 1e6:.0f} MB cap."
        )


def build_manifest(
    *,
    class_name: str,
    root: Path,
    files: List[Path],
    packager_version: str = "0.1.0",
) -> Dict[str, Any]:
    """packaging.py::build_manifest -- field-for-field.

    ``whestbench_version`` / ``flopscope_version`` come from installed
    distribution metadata; in this sandbox neither is pip-installed, so both
    read ``"unknown"`` -- which is exactly what the real ``build_manifest``
    would write here.  ``validate_package`` does not check them.
    """
    non_files = [p for p in files if not p.is_file()]
    if non_files:
        offending = ", ".join(sorted(str(p.relative_to(root)) for p in non_files))
        raise ValueError(
            f"Refusing to build a manifest: every files[] entry must be a regular "
            f"file, but these are a directory or special file: {offending}."
        )
    manifest_files = [{"name": str(p.relative_to(root)), "sha256": _sha256(p)} for p in files]
    import numpy as np  # packager-side only; never reaches the submission

    return {
        "schema_version": "1.0",
        "api_version": "2.0",
        "entrypoint": {"module": "estimator", "class": class_name},
        "python": {
            "min_version": (
                f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
            )
        },
        "files": manifest_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packager_version": packager_version,
        "whestbench_version": _installed_version("whestbench"),
        "flopscope_version": _installed_version("flopscope"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def resolve_entrypoint_class(entry: Path) -> str:
    """Replicate loader.py::_resolve_estimator_class + packaging._validate_predict_signature.

    ``whestbench`` is not installed here, so a minimal stand-in supplies
    ``BaseEstimator`` (the grader supplies the real one).  Resolution order is
    the loader's: a class literally named ``Estimator`` wins; otherwise the
    unique ``BaseEstimator`` subclass defined in the module.
    """
    import importlib.util
    import inspect
    import types

    # Packaging must not mutate the folder it is packaging.
    sys.dont_write_bytecode = True
    created_stub = "whestbench" not in sys.modules
    if created_stub:
        import abc

        wb = types.ModuleType("whestbench")

        class BaseEstimator(abc.ABC):
            @abc.abstractmethod
            def predict(self, mlp, budget):  # noqa: ANN001
                raise NotImplementedError

            def setup(self, context) -> None:  # noqa: ANN001
                return None

            def teardown(self) -> None:
                return None

        wb.BaseEstimator = BaseEstimator
        sys.modules["whestbench"] = wb
    base = sys.modules["whestbench"].BaseEstimator

    sub_dir = str(entry.parent)
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    spec = importlib.util.spec_from_file_location("_whest_pack_probe", entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_whest_pack_probe"] = module
    spec.loader.exec_module(module)

    candidates = [
        v
        for v in vars(module).values()
        if inspect.isclass(v)
        and issubclass(v, base)
        and v is not base
        and v.__module__ == module.__name__
    ]
    named = [c for c in candidates if c.__name__ == "Estimator"]
    if named:
        cls = named[0]
    elif len(candidates) == 1:
        cls = candidates[0]
    elif not candidates:
        raise ValueError(f"No BaseEstimator subclasses found in {entry}.")
    else:
        raise ValueError(
            f"Ambiguous estimator classes in {entry}: "
            + ", ".join(c.__name__ for c in candidates)
        )

    # packaging.py::_validate_predict_signature -- the grader calls predict(mlp, budget).
    try:
        sig = inspect.signature(cls().predict)
    except (TypeError, ValueError):
        pass
    else:
        try:
            sig.bind(None, None)
        except TypeError as e:
            raise ValueError(
                f"Estimator.predict has an incompatible signature: the grader calls "
                f"predict(mlp, budget), but binding two positional arguments failed ({e})."
            ) from e

    sys.modules.pop("_whest_pack_probe", None)
    return cls.__name__


def write_archive(*, root: Path, bundled: List[Path], manifest: Dict[str, Any], target: Path) -> Path:
    """packaging.py::_finalize_package -- the archive-writing half.

    Bundled files keep their real mode/mtime (``TarFile.add``); ``manifest.json``
    is synthesised last with ``mtime = now``.  The archive is therefore NOT
    byte-reproducible run to run (tar stores mtimes, gzip stores a timestamp) --
    exactly like ``whest package``.  The *contents* are deterministic and that
    is what ``validate_package`` and the grader check.
    """
    arcnames = {str(p.relative_to(root)) for p in bundled}
    entry_file = f"{manifest['entrypoint']['module']}.py"
    if entry_file not in arcnames:
        raise ValueError(
            f"Refusing to package: manifest declares entrypoint module "
            f"{manifest['entrypoint']['module']!r} (the grader imports {entry_file}), "
            f"but {entry_file} is not in the bundle ({sorted(arcnames)})."
        )

    manifest_blob = json.dumps(manifest, indent=2).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for path in bundled:
                archive.add(path, arcname=str(path.relative_to(root)))
            tinfo = tarfile.TarInfo(name="manifest.json")
            tinfo.size = len(manifest_blob)
            tinfo.mtime = int(datetime.now(timezone.utc).timestamp())
            import io

            archive.addfile(tinfo, fileobj=io.BytesIO(manifest_blob))
    return target


# --------------------------------------------------------------------------
# STAGE 2 -- validation (whestbench/validation.py::validate_package)
# --------------------------------------------------------------------------

_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})
_SUPPORTED_API_VERSIONS = frozenset({"2.0"})


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    name: str
    message: str


@dataclass(frozen=True)
class PackageValidation:
    ok: bool
    issues: List[ValidationIssue]


def _directory_issue(name: str) -> ValidationIssue:
    return ValidationIssue(
        "directory_entry",
        name,
        f"manifest files[] entry {name!r} is a directory. A directory cannot be "
        f"hashed and crashes the grader's integrity check.",
    )


def validate_package(tarball: "str | Path") -> PackageValidation:
    """validation.py::validate_package -- collects every issue, not just the first."""
    path = Path(tarball)
    if not path.is_file():
        return PackageValidation(
            False,
            [ValidationIssue("archive_missing", str(path), f"No such submission archive: {path}.")],
        )
    try:
        archive = tarfile.open(path, "r:gz")
    except (tarfile.TarError, OSError) as e:
        return PackageValidation(
            False,
            [
                ValidationIssue(
                    "archive_unreadable", str(path), f"Cannot open {path} as a gzip tar: {e}."
                )
            ],
        )

    issues: List[ValidationIssue] = []
    with archive:
        members = {m.name: m for m in archive.getmembers()}

        manifest_member = members.get("manifest.json")
        if manifest_member is None or not manifest_member.isreg():
            return PackageValidation(
                False,
                [
                    ValidationIssue(
                        "missing_manifest",
                        "manifest.json",
                        "Archive has no readable manifest.json.",
                    )
                ],
            )
        try:
            manifest = json.loads(archive.extractfile(manifest_member).read())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return PackageValidation(
                False,
                [
                    ValidationIssue(
                        "invalid_manifest_json", "manifest.json", f"manifest.json is not valid JSON: {e}."
                    )
                ],
            )
        if not isinstance(manifest, dict):
            return PackageValidation(
                False,
                [
                    ValidationIssue(
                        "invalid_manifest_json",
                        "manifest.json",
                        f"manifest.json must be a JSON object, not a {type(manifest).__name__}.",
                    )
                ],
            )

        if manifest.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
            issues.append(
                ValidationIssue(
                    "unsupported_schema_version",
                    "manifest.json",
                    f"manifest schema_version {manifest.get('schema_version')!r} is not supported.",
                )
            )
        if manifest.get("api_version") not in _SUPPORTED_API_VERSIONS:
            issues.append(
                ValidationIssue(
                    "unsupported_api_version",
                    "manifest.json",
                    f"manifest api_version {manifest.get('api_version')!r} is not supported.",
                )
            )

        entrypoint = manifest.get("entrypoint")
        if not isinstance(entrypoint, dict):
            entrypoint = {}
        module = entrypoint.get("module")
        if not module:
            issues.append(
                ValidationIssue(
                    "missing_entrypoint", "manifest.json", "manifest entrypoint.module is missing."
                )
            )
        else:
            entry_file = f"{module}.py"
            m = members.get(entry_file)
            if m is None or not m.isreg():
                issues.append(
                    ValidationIssue(
                        "missing_entrypoint_file",
                        entry_file,
                        f"manifest declares entrypoint module {module!r} but the archive "
                        f"has no regular file {entry_file}.",
                    )
                )

        files = manifest.get("files")
        if not isinstance(files, list):
            issues.append(
                ValidationIssue(
                    "invalid_files", "manifest.json", "manifest files[] is missing or not a list."
                )
            )
            files = []

        for entry in files:
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            declared = entry.get("sha256") if isinstance(entry, dict) else None
            if not name:
                issues.append(
                    ValidationIssue(
                        "invalid_files_entry", "", f"manifest files[] entry with no name: {entry!r}."
                    )
                )
                continue
            if name.endswith("/"):
                issues.append(_directory_issue(name))
                continue
            m = members.get(name)
            if m is None:
                issues.append(
                    ValidationIssue(
                        "file_not_in_archive",
                        name,
                        f"manifest lists {name!r} but the archive does not contain it.",
                    )
                )
                continue
            if m.isdir():
                issues.append(_directory_issue(name))
                continue
            if not m.isreg():
                issues.append(
                    ValidationIssue(
                        "not_a_regular_file",
                        name,
                        f"manifest entry {name!r} is not a regular file in the archive.",
                    )
                )
                continue
            handle = archive.extractfile(m)
            if handle is None:
                issues.append(
                    ValidationIssue(
                        "not_a_regular_file", name, f"manifest entry {name!r} could not be read."
                    )
                )
                continue
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
            actual = digest.hexdigest()
            if declared != actual:
                issues.append(
                    ValidationIssue(
                        "sha256_mismatch",
                        name,
                        f"sha256 mismatch for {name!r}: manifest says {declared}, "
                        f"archive bytes hash to {actual}.",
                    )
                )

    return PackageValidation(not issues, issues)


# --------------------------------------------------------------------------
# STAGE 3 -- surface audit
# --------------------------------------------------------------------------

# Things that must never appear in a file that ships to a public leaderboard.
_CREDENTIAL_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"AICROWD_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}", "hardcoded AICROWD_API_KEY value"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "PEM private key block"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
    (r"\bASIA[0-9A-Z]{16}\b", "AWS temporary access key id"),
    (r"aws_secret_access_key\s*=", "AWS secret access key assignment"),
    (r"\bghp_[A-Za-z0-9]{30,}\b", "GitHub personal access token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "GitHub fine-grained token"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "Slack token"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "OpenAI-style secret key"),
    (r"Authorization\s*:\s*Token\s+[A-Za-z0-9]{8,}", "literal Authorization: Token header"),
    (r"\bBEGIN OPENSSH PRIVATE KEY\b", "OpenSSH private key"),
    (r"password\s*[:=]\s*['\"][^'\"]{6,}['\"]", "hardcoded password literal"),
)

# Path fragments that mean we are shipping local scratch state.
_FORBIDDEN_PATH_FRAGMENTS = ("__pycache__", "_artifacts", ".git", ".venv", "_scratch", ".ipynb_checkpoints")


def audit_archive(tarball: Path, *, source_root: Path) -> Dict[str, Any]:
    """Report exactly what leaves the machine, and refuse anything that must not."""
    rows: List[Dict[str, Any]] = []
    with tarfile.open(tarball, "r:gz") as archive:
        for m in sorted(archive.getmembers(), key=lambda x: x.name):
            rows.append(
                {
                    "name": m.name,
                    "size": m.size,
                    "mode": oct(m.mode),
                    "type": ("reg" if m.isreg() else "dir" if m.isdir() else "other"),
                }
            )
        total = sum(r["size"] for r in rows)

        # 1. every member must be a regular file (no dirs, no symlinks, no devices)
        for r in rows:
            if r["type"] != "reg":
                bad(f"archive member {r['name']!r} is a {r['type']}, not a regular file")

        # 2. no path escapes, no absolute paths
        for r in rows:
            n = r["name"]
            if n.startswith("/") or ".." in Path(n).parts:
                bad(f"archive member {n!r} has an unsafe path")

        # 3. no scratch/VCS/artifact state
        for r in rows:
            for frag in _FORBIDDEN_PATH_FRAGMENTS:
                if frag in Path(r["name"]).parts or r["name"].endswith(".pyc"):
                    bad(f"archive member {r['name']!r} contains forbidden path fragment {frag!r}")

        # 4. no credential-shaped filenames
        for r in rows:
            if _is_secret(Path(r["name"])):
                bad(f"archive member {r['name']!r} matches a _SECRET_IGNORES pattern")

        # 5. content scan of every text member
        for m in archive.getmembers():
            if not m.isreg():
                continue
            fh = archive.extractfile(m)
            if fh is None:
                continue
            blob = fh.read()
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                continue  # binary (corrector.npz); checked structurally below
            for pat, label in _CREDENTIAL_PATTERNS:
                if re.search(pat, text):
                    bad(f"{m.name}: matches credential pattern -- {label}")

    # 6. the npz payload must be plain arrays, no pickled objects
    with tarfile.open(tarball, "r:gz") as archive:
        for m in archive.getmembers():
            if m.name.endswith(".npz") and m.isreg():
                import zipfile
                import io as _io

                data = archive.extractfile(m).read()
                try:
                    with zipfile.ZipFile(_io.BytesIO(data)) as z:
                        names = z.namelist()
                        if any(not n.endswith(".npy") for n in names):
                            bad(f"{m.name}: contains non-.npy members {names}")
                        else:
                            ok(f"{m.name}: {len(names)} plain .npy arrays ({', '.join(names)})")
                except zipfile.BadZipFile:
                    bad(f"{m.name}: not a readable npz/zip container")

    # 7. what the packager deliberately dropped, so it is on the record
    dropped_secrets = find_secret_files(source_root)
    if dropped_secrets:
        warn(
            "credential-shaped files present in the source folder and excluded: "
            + ", ".join(str(p.relative_to(source_root)) for p in dropped_secrets)
        )
    else:
        ok("no credential-shaped files anywhere under the source folder")

    # 8. licensing: everything shipped is first-party under the repo's license
    lic = REPO / "LICENSE"
    if lic.is_file():
        first_line = lic.read_text(encoding="utf-8").splitlines()[0].strip()
        ok(f"repo LICENSE present ({first_line}); all shipped files are first-party")
    else:
        warn("no LICENSE file at the repo root")

    return {"members": rows, "total_bytes": total}


def static_import_scan(tarball: Path) -> Dict[str, List[str]]:
    """Every top-level module name imported anywhere in any shipped .py file.

    AST-based, so it also sees imports nested inside functions and branches the
    runtime probe never takes -- a lazy ``import numpy`` in a rarely hit
    fallback would pass a runtime-only check and then kill a grader worker.
    """
    names: set = set()
    with tarfile.open(tarball, "r:gz") as archive:
        for m in archive.getmembers():
            if not (m.isreg() and m.name.endswith(".py")):
                continue
            src = archive.extractfile(m).read().decode("utf-8")
            tree = ast.parse(src, filename=m.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        names.add(node.module.split(".")[0])
                    elif node.level > 0:
                        names.add(f"<relative:{node.module or ''}>")
    return {"imports": sorted(names)}


# --------------------------------------------------------------------------
# STAGE 4 -- sandbox isolation + runtime proof
# --------------------------------------------------------------------------

# What the grader sandbox provides: flopscope, the whestbench API, a reduced stdlib.
_SANDBOX_ALLOW = {
    "__future__",
    "flopscope",
    "whestbench",
    # reduced stdlib -- conservative; anything outside this warns rather than fails
    "abc", "array", "bisect", "collections", "contextlib", "copy", "dataclasses",
    "decimal", "enum", "fractions", "functools", "hashlib", "heapq", "io",
    "itertools", "json", "math", "operator", "os", "pathlib", "random", "re",
    "statistics", "string", "struct", "sys", "textwrap", "time", "types",
    "typing", "warnings", "zipfile",
}

# Hard-denied: present in this dev box, absent on the grader. An accidental
# import of any of these passes locally and dies in the eval worker.
_SANDBOX_DENY = {
    "numpy", "scipy", "torch", "pandas", "sklearn", "jax", "jaxlib", "numba",
    "cupy", "tensorflow", "matplotlib", "opt_einsum", "rich", "datasets",
    "httpx", "requests", "toml", "whestfloor", "scripts", "tests",
}

_PROBE_SOURCE = r'''
"""Fresh-interpreter sandbox probe. Run with the repo OFF sys.path.

argv: <extracted_submission_dir> <json_config>
Emits one JSON line per phase on stdout, prefixed with "@@".
"""
import builtins, importlib.util, json, os, sys, time, types

sys.dont_write_bytecode = True  # keep the extracted artifact pristine

SUB = sys.argv[1]
CFG = json.loads(sys.argv[2])
ALLOW = set(CFG["allow"])
DENY = set(CFG["deny"])
WIDTH, DEPTH = CFG["width"], CFG["depth"]
BUDGET = CFG["budget"]
LAM = CFG["lam"]
WALL = CFG["wall"]


def emit(phase, **kw):
    kw["phase"] = phase
    sys.stdout.write("@@" + json.dumps(kw) + "\n")
    sys.stdout.flush()


# --- 0. isolation controls -------------------------------------------------
# The repo must be unreachable: a stray `import whestfloor` has to fail loudly.
repo_reachable = importlib.util.find_spec("whestfloor") is not None
emit("isolation", whestfloor_importable=repo_reachable,
     sys_path=[p for p in sys.path], cwd=os.getcwd())
if repo_reachable:
    emit("fatal", error="whestfloor is importable: the repo is still on sys.path")
    raise SystemExit(3)

# --- 1. the sandbox's contents --------------------------------------------
# The real worker imports flopscope + whestbench before touching participant
# code; do the same so the setup clock covers the same imports.
import flopscope as flops
import flopscope.numpy as fnp

# whestbench is not pip-installable here (pypi 403). The grader supplies the
# real package; this stand-in provides exactly the two names the contract
# exposes to an estimator.
if "whestbench" not in sys.modules:
    import abc
    from dataclasses import dataclass
    from typing import Optional

    wb = types.ModuleType("whestbench")

    class BaseEstimator(abc.ABC):
        @abc.abstractmethod
        def predict(self, mlp, budget):
            raise NotImplementedError

        def setup(self, context):
            return None

        def teardown(self):
            return None

    @dataclass(frozen=True)
    class SetupContext:
        width: int
        depth: int
        flop_budget: int
        api_version: str
        scratch_dir: Optional[str] = None
        submission_dir: Optional[str] = None
        seed: int = 0

    wb.BaseEstimator = BaseEstimator
    wb.SetupContext = SetupContext
    sys.modules["whestbench"] = wb

BaseEstimator = sys.modules["whestbench"].BaseEstimator
SetupContext = sys.modules["whestbench"].SetupContext

# --- 2. the import firewall -----------------------------------------------
# Installed into the estimator module's OWN globals, so it intercepts every
# import statement executed by submission code -- including cached modules and
# lazy imports inside functions -- while leaving flopscope's internals alone.
observed = []
violations = []
outside = []
_real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0:
        top = name.split(".")[0]
        observed.append(name)
        if top in DENY:
            violations.append(top)
            raise ImportError(
                "SANDBOX: %r is not available in the grader environment "
                "(only flopscope, whestbench and a reduced stdlib are)." % top
            )
        if top not in ALLOW:
            outside.append(top)
    return _real_import(name, globals, locals, fromlist, level)


guarded_builtins = dict(vars(builtins))
guarded_builtins["__import__"] = guarded_import

# positive control: the firewall must actually bite
try:
    exec("import numpy", {"__builtins__": guarded_builtins})
except ImportError as e:
    firewall_live = "SANDBOX:" in str(e)
else:
    firewall_live = False
observed.clear()
violations.clear()
outside.clear()
emit("firewall", live=firewall_live)
if not firewall_live:
    emit("fatal", error="import firewall did not block a deliberate `import numpy`")
    raise SystemExit(4)

# --- 3. load estimator.py the way the grader's loader does -----------------
# loader.py puts the estimator's own directory on sys.path so sibling modules
# resolve, then imports the file by path.
if SUB not in sys.path:
    sys.path.insert(0, SUB)

entry = os.path.join(SUB, "estimator.py")
spec = importlib.util.spec_from_file_location("_whestbench_submission_probe", entry)
module = importlib.util.module_from_spec(spec)
module.__dict__["__builtins__"] = guarded_builtins  # exec() will not overwrite this
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# the manifest entrypoint module name must also resolve as a plain `estimator`
plain = importlib.util.find_spec("estimator")
plain_ok = plain is not None and os.path.realpath(plain.origin) == os.path.realpath(entry)

candidates = [
    v for v in vars(module).values()
    if isinstance(v, type) and issubclass(v, BaseEstimator)
    and v is not BaseEstimator and v.__module__ == module.__name__
]
named = [c for c in candidates if c.__name__ == "Estimator"]
cls = named[0] if named else (candidates[0] if len(candidates) == 1 else None)
if cls is None:
    emit("fatal", error="could not resolve a unique Estimator class")
    raise SystemExit(5)

est = cls()
ctx = SetupContext(
    width=WIDTH, depth=DEPTH, flop_budget=BUDGET, api_version=CFG["api_version"],
    scratch_dir=None, submission_dir=SUB, seed=0,
)
est.setup(ctx)

# The 5 s setup window closes here: the parent stamps the arrival of this line,
# so the measurement spans interpreter start + imports + load + setup().
emit(
    "setup_done",
    cls=cls.__name__,
    plain_import_resolves=plain_ok,
    imports=sorted(set(observed)),
    violations=sorted(set(violations)),
    outside_allowlist=sorted(set(outside)),
    beta_loaded=getattr(est, "_beta", "n/a") is not None,
)

# --- 4. predict on a real MLP ---------------------------------------------
# Harness-side numpy is fine: the restriction applies to the estimator's
# globals, which is where the firewall is installed.
import numpy as _np

rng = _np.random.default_rng(0)
Wn = [
    (rng.standard_normal((WIDTH, WIDTH)) * (2.0 / WIDTH) ** 0.5).astype(_np.float32)
    for _ in range(DEPTH)
]
W = [fnp.asarray(w) for w in Wn]


class _MLP:
    def __init__(self, weights, seed=0):
        self.width = WIDTH
        self.depth = DEPTH
        self.weights = weights
        self.seed = seed
        self.name = "sandbox-probe"


t0 = time.perf_counter()
with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=WALL, quiet=True) as bctx:
    out = est.predict(_MLP(W), BUDGET)
    flops_used = bctx.flops_used
predict_wall = time.perf_counter() - t0

arr = _np.asarray(out)
residual = bctx.residual_wall_time_s or 0.0
C = float(flops_used) + LAM * float(residual)

emit(
    "predict_done",
    shape=list(arr.shape),
    dtype=str(arr.dtype),
    finite=bool(_np.isfinite(arr).all()),
    min=float(arr.min()),
    max=float(arr.max()),
    flops_used=int(flops_used),
    flop_ratio=float(flops_used) / BUDGET,
    residual_wall_time_s=float(residual),
    backend_time_s=float(bctx.flopscope_backend_time_s or 0.0),
    wall_time_s=float(bctx.wall_time_s or 0.0),
    predict_wall_s=predict_wall,
    effective_compute=C,
    compute_ratio=C / BUDGET,
    imports=sorted(set(observed)),
    violations=sorted(set(violations)),
    outside_allowlist=sorted(set(outside)),
)
est.teardown()
emit("done")
'''


def _sandbox_env() -> Dict[str, str]:
    """Child environment with the repo scrubbed off PYTHONPATH.

    flopscope itself lives on PYTHONPATH in this sandbox, so PYTHONPATH cannot
    simply be dropped (and ``-I``/``-E`` would drop it).  Every entry that
    resolves inside the repo is removed instead, and the child runs with ``-P``
    so neither the cwd nor the script directory is prepended either.
    """
    env = dict(os.environ)
    keep = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved == REPO or REPO in resolved.parents:
            continue
        keep.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(keep)
    env.pop("PYTHONSTARTUP", None)
    return env


def _time_interpreter_baseline(env: Dict[str, str], workdir: Path) -> float:
    """Wall time for a bare interpreter that only imports flopscope."""
    code = "import flopscope, flopscope.numpy, sys; sys.stdout.write('x'); sys.stdout.flush()"
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-P", "-c", code],
            env=env, cwd=str(workdir), capture_output=True, timeout=120,
        )
        best = min(best, time.perf_counter() - t0)
    return best


def run_sandbox_probe(tarball: Path, *, keep: bool = False) -> Dict[str, Any]:
    """Extract, then load + setup + predict in a fresh, repo-free interpreter."""
    workdir = Path(tempfile.mkdtemp(prefix="whest-sandbox-"))
    extract = workdir / "submission"
    extract.mkdir()
    with tarfile.open(tarball, "r:gz") as archive:
        for m in archive.getmembers():
            if not m.isreg():
                raise ValueError(f"refusing to extract non-regular member {m.name!r}")
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError(f"refusing to extract unsafe path {m.name!r}")
        archive.extractall(extract, filter="data")

    probe = workdir / "sandbox_probe.py"
    probe.write_text(_PROBE_SOURCE, encoding="utf-8")

    cfg = json.dumps(
        {
            "allow": sorted(_SANDBOX_ALLOW),
            "deny": sorted(_SANDBOX_DENY),
            "width": WIDTH,
            "depth": DEPTH,
            "budget": FLOP_BUDGET,
            "lam": LAMBDA_FLOPS_PER_SECOND,
            "wall": WALL_TIME_LIMIT_S,
            "api_version": API_VERSION,
        }
    )
    env = _sandbox_env()
    info(f"child PYTHONPATH = {env['PYTHONPATH'] or '(empty)'}")
    info(f"child cwd        = {workdir}")

    baseline = _time_interpreter_baseline(env, workdir)

    events: List[Dict[str, Any]] = []
    stamps: Dict[str, float] = {}
    # The window the grader enforces opens when the worker process is spawned
    # (SubprocessRunner.start: Popen, send "start", read the reply within
    # setup_timeout_s) and closes when setup() answers. Stamp it the same way.
    t_spawn = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-P", str(probe), str(extract), cfg],
        env=env,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("@@"):
            ev = json.loads(line[2:])
            stamps[ev["phase"]] = time.perf_counter() - t_spawn
            events.append(ev)
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()

    result = {
        "returncode": rc,
        "events": {e["phase"]: e for e in events},
        "stamps": stamps,
        "stderr": stderr,
        "interpreter_baseline_s": baseline,
        "extract_dir": str(extract),
    }
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        info(f"kept extraction at {extract}")
    return result


def report_sandbox(res: Dict[str, Any]) -> None:
    ev = res["events"]
    st = res["stamps"]

    if res["returncode"] != 0 or "done" not in ev:
        bad(f"sandbox probe exited {res['returncode']}")
        if ev.get("fatal"):
            bad(f"probe fatal: {ev['fatal'].get('error')}")
        if res["stderr"].strip():
            for ln in res["stderr"].strip().splitlines()[-25:]:
                info(f"stderr| {ln}")
        return

    iso = ev.get("isolation", {})
    if iso.get("whestfloor_importable"):
        bad("repo IS importable from the sandbox -- isolation is not real")
    else:
        ok("repo is not importable from the sandbox (`import whestfloor` -> ModuleNotFoundError)")
    info(f"child sys.path = {iso.get('sys_path')}")

    if ev.get("firewall", {}).get("live"):
        ok("import firewall verified live (a deliberate `import numpy` was blocked)")
    else:
        bad("import firewall did not engage")

    s = ev.get("setup_done", {})
    ok(f"Estimator class resolved: {s.get('cls')}")
    if s.get("plain_import_resolves"):
        ok("manifest entrypoint module `estimator` resolves to the extracted estimator.py")
    else:
        bad("`import estimator` does not resolve to the extracted file")
    if s.get("beta_loaded"):
        ok("setup() loaded corrector.npz (estimator._beta is populated)")
    else:
        bad("setup() did NOT load corrector.npz -- the head would silently degrade")

    setup_s = st.get("setup_done", float("nan"))
    line = (
        f"setup wall time {setup_s:.3f}s against the {SETUP_TIMEOUT_S:.1f}s cap "
        f"(spawn -> setup() returned; includes interpreter start + flopscope import)"
    )
    if setup_s < SETUP_TIMEOUT_S:
        ok(line)
    else:
        bad(line)
    delta = setup_s - res["interpreter_baseline_s"]
    info(
        f"bare interpreter + flopscope import alone: {res['interpreter_baseline_s']:.3f}s "
        f"-> load + setup() adds {delta:+.3f}s"
        + (" (below run-to-run noise)" if abs(delta) < 0.05 else "")
    )
    info(
        "the real worker also imports the whestbench package inside this window, "
        "which this box cannot install; treat the number as a lower bound"
    )

    p = ev.get("predict_done", {})
    shape = tuple(p.get("shape", []))
    if shape == (DEPTH, WIDTH):
        ok(f"predict returned shape {shape} as required")
    else:
        bad(f"predict returned shape {shape}, expected {(DEPTH, WIDTH)}")
    if p.get("finite"):
        ok(f"all {DEPTH * WIDTH} predictions finite (range {p.get('min'):.6g} .. {p.get('max'):.6g})")
    else:
        bad("predictions contain non-finite values")

    cr = p.get("compute_ratio", float("nan"))
    if cr <= 1.0:
        ok(
            f"effective compute C = {p.get('effective_compute'):.4g} "
            f"= {cr * 100:.2f}% of B = {FLOP_BUDGET:.4g}"
        )
    else:
        bad(f"effective compute is {cr * 100:.1f}% of budget -- this MLP would be zeroed")
    info(
        f"F = {p.get('flops_used'):,} ({p.get('flop_ratio') * 100:.2f}% of B), "
        f"R = {p.get('residual_wall_time_s'):.4f}s -> lambda*R = "
        f"{LAMBDA_FLOPS_PER_SECOND * p.get('residual_wall_time_s', 0.0):.4g} FLOPs"
    )
    info(f"predict wall time {p.get('predict_wall_s'):.2f}s against the {WALL_TIME_LIMIT_S:.0f}s hard cap")

    imports = sorted(set(s.get("imports", [])) | set(p.get("imports", [])))
    viol = sorted(set(s.get("violations", [])) | set(p.get("violations", [])))
    out = sorted(set(s.get("outside_allowlist", [])) | set(p.get("outside_allowlist", [])))
    info(f"imports executed by submission code: {imports or '(none beyond module load)'}")
    if viol:
        bad(f"submission imported denied modules: {viol}")
    else:
        ok("no denied module (numpy / scipy / torch / whestfloor / ...) was imported at runtime")
    if out:
        warn(f"imports outside the conservative sandbox allowlist: {out}")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def default_out_dir() -> Path:
    art = os.environ.get("WHEST_ARTIFACTS")
    if art:
        return Path(art)
    return Path(tempfile.gettempdir())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(REPO / "submission"),
                    help="submission folder (default: <repo>/submission)")
    ap.add_argument("--out", default=None,
                    help="tarball path (default: $WHEST_ARTIFACTS/submission-<utc>.tar.gz)")
    ap.add_argument("--no-latest", action="store_true",
                    help="do not also write submission-latest.tar.gz")
    ap.add_argument("--skip-sandbox", action="store_true", help="skip the fresh-interpreter proof")
    ap.add_argument("--keep-extract", action="store_true", help="leave the extraction dir on disk")
    ap.add_argument("--verify-only", metavar="TARBALL", default=None,
                    help="validate + audit + sandbox-test an existing tarball; build nothing")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write a machine-readable report here "
                         "(default: $WHEST_ARTIFACTS/package_report.json)")
    args = ap.parse_args(argv)

    report: Dict[str, Any] = {"utc": datetime.now(timezone.utc).isoformat()}

    # ---- stage 1 ---------------------------------------------------------
    if args.verify_only:
        tarball = Path(args.verify_only).resolve()
        source_root = Path(args.source).resolve()
        head(f"STAGE 1  package  (skipped -- verifying {tarball})")
        ok(f"using existing archive {tarball} ({human(tarball.stat().st_size)})")
    else:
        head("STAGE 1  package")
        root, entry, mode = resolve_submission(args.source)
        source_root = root
        ok(f"resolved {args.source} -> root={root} entry={entry.name} mode={mode}")

        bundled = collect_submission_files(root)
        enforce_submission_caps(bundled)
        total = sum(p.stat().st_size for p in bundled)
        ok(f"{len(bundled)}/{MAX_SUBMISSION_FILES} files, {total} B / {MAX_SUBMISSION_BYTES} B "
           f"({human(total)})")
        for p in bundled:
            info(f"+ {p.relative_to(root)}  {p.stat().st_size} B")
        skipped = [
            p.relative_to(root)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p not in bundled
        ]
        for p in skipped:
            info(f"- {p}  (ignored)")
        if entry not in bundled:
            bad("estimator.py is excluded by an ignore pattern but must ship")
            return 1

        class_name = resolve_entrypoint_class(entry)
        ok(f"entrypoint class resolves to {class_name!r}; predict(mlp, budget) binds")

        manifest = build_manifest(class_name=class_name, root=root, files=bundled)
        ok(f"manifest: schema_version={manifest['schema_version']} "
           f"api_version={manifest['api_version']} entrypoint={manifest['entrypoint']}")
        for f in manifest["files"]:
            info(f"sha256 {f['sha256']}  {f['name']}")

        if args.out:
            tarball = Path(args.out).resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            tarball = (default_out_dir() / f"submission-{stamp}.tar.gz").resolve()
        write_archive(root=root, bundled=bundled, manifest=manifest, target=tarball)
        ok(f"wrote {tarball} ({tarball.stat().st_size} B / {human(tarball.stat().st_size)})")
        report["manifest"] = manifest
        report["source_files"] = [str(p.relative_to(root)) for p in bundled]

        if not args.no_latest and not args.out:
            latest = default_out_dir() / "submission-latest.tar.gz"
            shutil.copyfile(tarball, latest)
            ok(f"copied to stable path {latest}")
            report["latest_path"] = str(latest)

    report["tarball"] = str(tarball)
    report["tarball_bytes"] = tarball.stat().st_size

    # ---- stage 2 ---------------------------------------------------------
    head("STAGE 2  validate  (validation.py::validate_package, re-hashing every member)")
    vr = validate_package(tarball)
    if vr.ok:
        ok("archive passes the same validation `whest submit` runs before upload")
    else:
        for i in vr.issues:
            bad(f"[{i.code}] {i.name or 'archive'}: {i.message}")
    report["validation_ok"] = vr.ok
    report["validation_issues"] = [
        {"code": i.code, "name": i.name, "message": i.message} for i in vr.issues
    ]

    # ---- stage 3 ---------------------------------------------------------
    head("STAGE 3  audit  (what actually leaves the machine)")
    audit = audit_archive(tarball, source_root=source_root)
    print()
    print(f"        {'member':<24} {'bytes':>10}  {'mode':>6}  sha256")
    with tarfile.open(tarball, "r:gz") as archive:
        for m in sorted(archive.getmembers(), key=lambda x: x.name):
            h = hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            print(f"        {m.name:<24} {m.size:>10}  {oct(m.mode):>6}  {h}")
    print(f"        {'TOTAL (uncompressed)':<24} {audit['total_bytes']:>10}")
    print(f"        {'tarball on disk':<24} {tarball.stat().st_size:>10}")
    print()

    imports = static_import_scan(tarball)
    denied = sorted(set(imports["imports"]) & _SANDBOX_DENY)
    outside = sorted(set(imports["imports"]) - _SANDBOX_ALLOW - set(denied))
    ok(f"static import scan of every shipped .py: {imports['imports']}")
    if denied:
        bad(f"shipped code statically imports denied modules: {denied}")
    else:
        ok("no denied module appears in any import statement, on any code path")
    if outside:
        warn(f"static imports outside the conservative allowlist: {outside}")
    report["audit"] = audit
    report["static_imports"] = imports["imports"]

    # ---- stage 4 ---------------------------------------------------------
    if args.skip_sandbox:
        head("STAGE 4  sandbox  (skipped)")
    else:
        head("STAGE 4  sandbox  (fresh interpreter, repo off sys.path)")
        res = run_sandbox_probe(tarball, keep=args.keep_extract)
        report_sandbox(res)
        report["sandbox"] = {
            "returncode": res["returncode"],
            "stamps": res["stamps"],
            "events": res["events"],
            "interpreter_baseline_s": res["interpreter_baseline_s"],
        }

    # ---- verdict ---------------------------------------------------------
    head("VERDICT")
    report["failures"] = list(_FAILURES)
    report["warnings"] = list(_WARNINGS)
    json_path = Path(args.json_out) if args.json_out else default_out_dir() / "package_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for w in _WARNINGS:
        print(f"  warn  {w}")
    if _FAILURES:
        for f in _FAILURES:
            print(f"  FAIL  {f}")
        print(f"\n  NOT SHIPPABLE -- {len(_FAILURES)} failure(s). Report: {json_path}")
        return 1
    print(f"  SHIPPABLE   {tarball}")
    print(f"              {tarball.stat().st_size} bytes ({human(tarball.stat().st_size)})")
    print(f"              report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
