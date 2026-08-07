#!/usr/bin/env python
"""Submit a packaged estimator tarball to AIcrowd.

Replicates ``whest submit`` using only the standard library.  The bundled
``whest`` CLI cannot be installed in this sandbox (PyPI returns 403, so
``uv sync`` can never resolve ``httpx``), but the wire protocol is small and
fully specified by ``whestbench/src/whestbench/aicrowd_client.py``:

    1. GET  {rails}/submissions?challenge_id=<slug>
           -> {"url": ..., "fields": {...}}       presigned S3 POST
    2. POST <url>  multipart/form-data, all fields + the file, NO auth header
           -> the object key (``${filename}`` substituted locally)
    3. POST {rails}/submissions
           {"challenge_id": <slug>,
            "submission": {"description": ...},
            "submission_files": [{"submission_file_s3_key": <key>}]}
    4. GET  {rails}/submissions/<id>
           -> grading_status_cd, score, grading_message

Auth is ``Authorization: Token <key>`` (not ``Bearer``).

Credential handling
-------------------
The key is read from ``AICROWD_API_KEY`` or from a file **outside the
repository** (default ``~/.aicrowd_key``).  It is never printed, never written
anywhere, never placed on a command line, and never included in any error
message: :func:`_redact` scrubs it from every string this script emits.

Safety
------
Dry-run is the default.  ``--send`` is required to transmit anything.  A dry
run performs steps 0 and 1 (identity + registration + presign) so the
credential path and the tarball are both proven before a real submission, but
uploads nothing and creates nothing.

Usage
-----
    python scripts/36_submit.py --tarball PATH                    # dry run
    python scripts/36_submit.py --tarball PATH --send -m "note"   # real
    python scripts/36_submit.py --status 123456                   # poll
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CHALLENGE_SLUG = "arc-white-box-estimation-challenge-2026"
RAILS_BASE = os.environ.get("RAILS_HOST_URL", "https://www.aicrowd.com/api/v1")
AICROWD_BASE = os.environ.get("AICROWD_API_ENDPOINT", "https://api.aicrowd.com")

# Matches whestbench's SUBMIT_RETRY: transient statuses are retried, the rest
# fail immediately with the server's own message.
RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
BACKOFF = (2.0, 4.0, 8.0, 16.0)

MAX_SUBMISSION_BYTES = 50 * 1024 * 1024

_KEY: str = ""


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def load_key(key_file: Optional[str]) -> str:
    """Return the API key from the environment or a file outside the repo.

    Never logs it.  Refuses a key file inside the working tree, because a
    credential under a directory we routinely ``git add`` is one ``git add -A``
    away from a public commit.
    """
    env = os.environ.get("AICROWD_API_KEY", "").strip()
    if env:
        return env
    path = Path(key_file or os.path.expanduser("~/.aicrowd_key")).expanduser()
    if not path.is_file():
        raise SystemExit(
            f"No API key. Set AICROWD_API_KEY, or write the key to {path} "
            f"(chmod 600). Get it from https://www.aicrowd.com/participants/me/edit"
        )
    repo = Path(__file__).resolve().parent.parent
    try:
        path.resolve().relative_to(repo)
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"Refusing to read a credential from inside the repository ({path}). "
            f"Move it outside the working tree, e.g. ~/.aicrowd_key."
        )
    key = path.read_text().strip()
    if not key:
        raise SystemExit(f"{path} is empty.")
    return key


def _redact(text: str) -> str:
    """Scrub the key from anything we are about to print."""
    if _KEY and _KEY in text:
        text = text.replace(_KEY, "<redacted>")
    return text


def say(msg: str) -> None:
    print(_redact(msg), flush=True)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _request(
    method: str,
    url: str,
    *,
    auth: bool = True,
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
    op: str = "request",
) -> Tuple[int, bytes]:
    """One logical request with bounded retries on transient failures.

    Returns ``(status, body)`` on success; raises ``SystemExit`` with the
    server's message on a permanent error.  The auth header is attached here
    and nowhere else, so the key never reaches a subprocess or a log line.
    """
    headers: Dict[str, str] = {"Accept": "application/json"}
    if auth:
        headers["Authorization"] = f"Token {_KEY}"
    if content_type:
        headers["Content-Type"] = content_type

    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:  # noqa: PERF203 - retry loop
            payload = exc.read()
            if exc.code not in RETRYABLE:
                detail = _server_message(payload) or payload[:400].decode(
                    "utf-8", "replace"
                )
                raise SystemExit(
                    _redact(f"{op} failed: HTTP {exc.code}. {detail}")
                ) from None
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = type(exc).__name__
        if attempt >= MAX_ATTEMPTS:
            raise SystemExit(_redact(f"{op} failed after {attempt} attempts ({last})."))
        delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
        say(f"  {op}: {last}; retrying in {delay:.0f}s "
            f"(attempt {attempt + 1}/{MAX_ATTEMPTS})")
        time.sleep(delay)
    raise SystemExit(f"{op}: unreachable")  # pragma: no cover


def _server_message(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - non-JSON error bodies are common
        return ""
    for key in ("message", "error", "detail", "errors"):
        if key in data:
            return str(data[key])
    return ""


def get_json(url: str, *, op: str) -> Any:
    _, payload = _request("GET", url, op=op)
    return json.loads(payload.decode("utf-8"))


def post_json(url: str, obj: Any, *, op: str) -> Any:
    _, payload = _request(
        "POST",
        url,
        body=json.dumps(obj).encode("utf-8"),
        content_type="application/json",
        op=op,
    )
    return json.loads(payload.decode("utf-8")) if payload else {}


def _multipart(fields: Dict[str, str], filename: str, blob: bytes) -> Tuple[bytes, str]:
    """Encode a presigned-POST body.

    S3 requires every policy field to precede the ``file`` part; anything after
    it is ignored and the signature check fails.  So the file goes last.
    """
    boundary = "----whestfloor" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += str(value).encode("utf-8") + b"\r\n"
    out += f"--{boundary}\r\n".encode()
    out += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/gzip\r\n\r\n"
    ).encode()
    out += blob + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
def verify_identity() -> int:
    data = get_json(f"{RAILS_BASE}/api_user", op="verifying your API key")
    return int(data["id"])


def resolve_challenge(slug: str) -> int:
    data = get_json(
        f"{AICROWD_BASE}/challenges/?{urllib.parse.urlencode({'slug': slug})}",
        op="resolving the challenge",
    )
    items = data if isinstance(data, list) else data.get("data", [])
    for item in items:
        if item.get("slug") == slug:
            return int(item["id"])
    if items:
        return int(items[0]["id"])
    raise SystemExit(f"challenge not found: {slug}")


def check_registration(challenge_id: int, participant_id: int) -> bool:
    q = urllib.parse.urlencode({"participant_id": participant_id})
    data = get_json(
        f"{AICROWD_BASE}/challenges/{challenge_id}/participant?{q}",
        op="checking your challenge registration",
    )
    return bool(data.get("registered"))


def get_upload_details(slug: str) -> Dict[str, Any]:
    q = urllib.parse.urlencode({"challenge_id": slug})
    data = get_json(f"{RAILS_BASE}/submissions?{q}", op="preparing the upload")
    return data.get("data", data)


def upload_to_s3(upload: Dict[str, Any], tarball: Path) -> str:
    fields = dict(upload["fields"])
    s3_key = fields.get("key", "").replace("${filename}", tarball.name)
    fields["key"] = s3_key
    body, ctype = _multipart(fields, tarball.name, tarball.read_bytes())
    _request(
        "POST",
        upload["url"],
        auth=False,  # the AIcrowd token must not travel to S3
        body=body,
        content_type=ctype,
        op="uploading the artifact",
    )
    return s3_key


def create_submission(slug: str, s3_key: str, description: str) -> Dict[str, Any]:
    return post_json(
        f"{RAILS_BASE}/submissions",
        {
            "challenge_id": slug,
            "submission": {"description": description},
            "submission_files": [{"submission_file_s3_key": s3_key}],
        },
        op="creating your submission",
    )


def get_status(submission_id: int) -> Dict[str, Any]:
    return get_json(
        f"{RAILS_BASE}/submissions/{submission_id}", op="checking submission status"
    )


def _report(st: Dict[str, Any]) -> None:
    say(
        f"  status={st.get('grading_status_cd')}  score={st.get('score')}  "
        f"score_secondary={st.get('score_secondary')}"
    )
    msg = st.get("grading_message")
    if msg:
        say(f"  message: {str(msg)[:800]}")


def watch(submission_id: int, timeout_s: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    delay = 10.0
    while True:
        st = get_status(submission_id)
        state = str(st.get("grading_status_cd") or "")
        if state in {"graded", "failed"} or time.monotonic() > deadline:
            return st
        say(f"  [{state or 'pending'}] waiting {delay:.0f}s ...")
        time.sleep(delay)
        delay = min(delay * 1.5, 60.0)


# --------------------------------------------------------------------------
def main() -> int:
    global _KEY

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tarball", help="packaged submission (scripts/35_package.py)")
    ap.add_argument(
        "--send",
        action="store_true",
        help="actually upload and create the submission (default is a dry run)",
    )
    ap.add_argument("-m", "--message", default="", help="submission description")
    ap.add_argument("--key-file", default=None, help="default ~/.aicrowd_key")
    ap.add_argument("--status", type=int, help="poll an existing submission id")
    ap.add_argument("--watch", action="store_true", help="poll until graded")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    _KEY = load_key(args.key_file)

    if args.status:
        st = get_status(args.status)
        _report(st)
        if args.watch:
            _report(watch(args.status, args.timeout))
        return 0

    if not args.tarball:
        ap.error("--tarball is required (or use --status)")
    tarball = Path(args.tarball).expanduser().resolve()
    if not tarball.is_file():
        raise SystemExit(f"no such tarball: {tarball}")
    size = tarball.stat().st_size
    if size > MAX_SUBMISSION_BYTES:
        raise SystemExit(
            f"tarball is {size:,} B, over the {MAX_SUBMISSION_BYTES:,} B cap"
        )

    say(f"tarball  {tarball}  ({size:,} B)")

    pid = verify_identity()
    say(f"identity participant_id={pid}")

    cid = resolve_challenge(CHALLENGE_SLUG)
    if not check_registration(cid, pid):
        raise SystemExit(
            f"participant {pid} is not registered for {CHALLENGE_SLUG}. "
            f"Accept the rules on the challenge page first."
        )
    say(f"challenge id={cid}  registered=True")

    upload = get_upload_details(CHALLENGE_SLUG)
    if "url" not in upload or "fields" not in upload:
        raise SystemExit(f"unexpected presign response: {sorted(upload)}")
    say(f"presign  host={urllib.parse.urlparse(upload['url']).netloc}  "
        f"fields={len(upload['fields'])}")

    if not args.send:
        say("")
        say("DRY RUN — nothing uploaded, nothing created.")
        say("Everything up to and including the presigned upload succeeded.")
        say("Re-run with --send to submit.")
        return 0

    s3_key = upload_to_s3(upload, tarball)
    say(f"uploaded s3_key={s3_key}")

    desc = args.message or f"whest-floor {tarball.name}"
    created = create_submission(CHALLENGE_SLUG, s3_key, desc)
    sub = created.get("data", created)
    sub_id = sub.get("id") or sub.get("submission_id")
    say(f"created  submission_id={sub_id}")
    print(json.dumps({"submission_id": sub_id, "s3_key": s3_key}, indent=2))

    if sub_id and args.watch:
        _report(watch(int(sub_id), args.timeout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
