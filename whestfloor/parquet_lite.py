"""A minimal, dependency-free reader for the subset of Parquet we need.

Why this exists
---------------
The official WhestBench ground truth ships as HuggingFace Parquet.  ``pyarrow``
cannot be installed here (no PyPI), and HF's ``datasets-server`` refuses the
file: its row groups are one-per-shard and ~309 MB, over the server's 300 MB
scan limit, so ``/rows`` and ``/first-rows`` both 500 with
``Scan size limit exceeded``.

But we do not want the bytes that make the file big.  99.4% of each shard is
the ``weights`` column, and weights are *derivable* from ``mlp_seed`` (see
:mod:`whestfloor.official_seeds`).  Parquet is a columnar format: the footer
gives the exact byte range of every column chunk, and HF serves ``Range:`` on
``/resolve/`` URLs.  So we read the 5.5 KB footer, then ~1.8 MB of the four
columns we actually want, and skip 307 MB per shard.

What is implemented
-------------------
Only what those four columns use, verified against the real file:

* Thrift **compact** protocol (``FileMetaData``, ``PageHeader``).
* **Snappy** raw block decompression.
* **RLE / bit-packed hybrid** (definition levels and dictionary indices).
* **PLAIN** and **RLE_DICTIONARY** data pages, v1, for INT64 / FLOAT / DOUBLE.

Anything else raises.  This is deliberately not a general Parquet reader; it is
a narrow tool that fails loudly rather than silently mis-decoding.
"""

from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Thrift compact protocol
# ---------------------------------------------------------------------------
_CT_TRUE, _CT_FALSE, _CT_BYTE = 1, 2, 3
_CT_I16, _CT_I32, _CT_I64, _CT_DOUBLE = 4, 5, 6, 7
_CT_BINARY, _CT_LIST, _CT_SET, _CT_MAP, _CT_STRUCT = 8, 9, 10, 11, 12


class _TCompact:
    """Cursor over a Thrift-compact buffer."""

    __slots__ = ("b", "p")

    def __init__(self, b: bytes, p: int = 0) -> None:
        self.b = b
        self.p = p

    def byte(self) -> int:
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self) -> int:
        r = s = 0
        while True:
            c = self.b[self.p]
            self.p += 1
            r |= (c & 0x7F) << s
            if not c & 0x80:
                return r
            s += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self) -> bytes:
        n = self.varint()
        v = self.b[self.p:self.p + n]
        self.p += n
        return bytes(v)

    def double(self) -> float:
        v = struct.unpack_from("<d", self.b, self.p)[0]
        self.p += 8
        return v


def _read_value(r: _TCompact, ct: int):
    if ct == _CT_TRUE:
        return True
    if ct == _CT_FALSE:
        return False
    if ct == _CT_BYTE:
        v = r.byte()
        return v - 256 if v > 127 else v
    if ct in (_CT_I16, _CT_I32, _CT_I64):
        return r.zigzag()
    if ct == _CT_DOUBLE:
        return r.double()
    if ct == _CT_BINARY:
        return r.binary()
    if ct in (_CT_LIST, _CT_SET):
        h = r.byte()
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n = r.varint()
        return [_read_value(r, et) for _ in range(n)]
    if ct == _CT_MAP:
        n = r.varint()
        if n == 0:
            return {}
        kv = r.byte()
        kt, vt = kv >> 4, kv & 0x0F
        return {_read_value(r, kt): _read_value(r, vt) for _ in range(n)}
    if ct == _CT_STRUCT:
        return read_struct(r)
    raise ValueError(f"unsupported thrift compact type {ct} at offset {r.p}")


def read_struct(r: _TCompact) -> dict:
    """Read a compact-protocol struct into ``{field_id: value}``."""
    out: dict[int, object] = {}
    last = 0
    while True:
        h = r.byte()
        if h == 0:
            return out
        delta, ct = h >> 4, h & 0x0F
        fid = r.zigzag() if delta == 0 else last + delta
        last = fid
        out[fid] = _read_value(r, ct)


# ---------------------------------------------------------------------------
# Snappy (raw block format)
# ---------------------------------------------------------------------------
def snappy_decompress(src: bytes) -> bytes:
    """Decode a raw Snappy block.  No framing, no CRC — that is what Parquet uses."""
    n = len(src)
    # preamble: varint uncompressed length
    i = 0
    ulen = shift = 0
    while True:
        c = src[i]
        i += 1
        ulen |= (c & 0x7F) << shift
        if not c & 0x80:
            break
        shift += 7
    out = bytearray(ulen)
    o = 0
    while i < n:
        tag = src[i]
        kind = tag & 0x03
        if kind == 0:  # literal
            ln = tag >> 2
            i += 1
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(src[i:i + nb], "little")
                i += nb
            ln += 1
            out[o:o + ln] = src[i:i + ln]
            i += ln
            o += ln
            continue
        if kind == 1:  # copy, 1-byte offset
            ln = 4 + ((tag >> 2) & 0x07)
            off = ((tag >> 5) << 8) | src[i + 1]
            i += 2
        elif kind == 2:  # copy, 2-byte offset
            ln = (tag >> 2) + 1
            off = int.from_bytes(src[i + 1:i + 3], "little")
            i += 3
        else:  # copy, 4-byte offset
            ln = (tag >> 2) + 1
            off = int.from_bytes(src[i + 1:i + 5], "little")
            i += 5
        s = o - off
        if off >= ln:
            out[o:o + ln] = out[s:s + ln]
            o += ln
        else:  # overlapping copy must be byte-by-byte
            for k in range(ln):
                out[o + k] = out[s + k]
            o += ln
    if o != ulen:
        raise ValueError(f"snappy: produced {o} bytes, header said {ulen}")
    return bytes(out)


# ---------------------------------------------------------------------------
# RLE / bit-packed hybrid
# ---------------------------------------------------------------------------
def _unpack_bits(buf: bytes, count: int, width: int) -> np.ndarray:
    if width == 0:
        return np.zeros(count, dtype=np.int64)
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    need = count * width
    if bits.size < need:
        raise ValueError("bit-packed run truncated")
    grid = bits[:need].reshape(count, width).astype(np.int64)
    return grid @ (np.int64(1) << np.arange(width, dtype=np.int64))


def rle_hybrid(buf: bytes, width: int, count: int) -> np.ndarray:
    """Decode ``count`` values of the RLE/bit-packed hybrid at ``buf[0:]``."""
    out = np.empty(count, dtype=np.int64)
    i = p = 0
    nb = (width + 7) // 8
    while i < count:
        hdr = shift = 0
        while True:
            c = buf[p]
            p += 1
            hdr |= (c & 0x7F) << shift
            if not c & 0x80:
                break
            shift += 7
        if hdr & 1:  # bit-packed run: (ngroups << 1) | 1, 8 values per group
            n = (hdr >> 1) * 8
            nbytes = (hdr >> 1) * width
            vals = _unpack_bits(buf[p:p + nbytes], n, width)
            p += nbytes
            take = min(n, count - i)
            out[i:i + take] = vals[:take]
        else:  # RLE run: (run_length << 1)
            n = hdr >> 1
            if n == 0:
                raise ValueError("zero-length RLE run")
            v = int.from_bytes(buf[p:p + nb], "little") if nb else 0
            p += nb
            take = min(n, count - i)
            out[i:i + take] = v
        i += take
    return out


# ---------------------------------------------------------------------------
# Parquet structures
# ---------------------------------------------------------------------------
PHYS = {0: "BOOLEAN", 1: "INT32", 2: "INT64", 3: "INT96", 4: "FLOAT",
        5: "DOUBLE", 6: "BYTE_ARRAY", 7: "FIXED_LEN_BYTE_ARRAY"}
CODEC = {0: "UNCOMPRESSED", 1: "SNAPPY", 2: "GZIP", 3: "LZO", 4: "BROTLI",
         5: "LZ4", 6: "ZSTD", 7: "LZ4_RAW"}
ENC = {0: "PLAIN", 2: "PLAIN_DICTIONARY", 3: "RLE", 4: "BIT_PACKED",
       5: "DELTA_BINARY_PACKED", 6: "DELTA_LENGTH_BYTE_ARRAY",
       7: "DELTA_BYTE_ARRAY", 8: "RLE_DICTIONARY", 9: "BYTE_STREAM_SPLIT"}
_NP = {"INT32": np.int32, "INT64": np.int64, "FLOAT": np.float32,
       "DOUBLE": np.float64}


@dataclass
class ColumnChunk:
    path: str
    physical: str
    codec: str
    num_values: int
    start: int       # first byte of the chunk (dictionary page if present)
    length: int      # total compressed bytes of the chunk
    dict_offset: int | None
    data_offset: int
    max_def: int
    max_rep: int


@dataclass
class FileMeta:
    num_rows: int
    created_by: str
    columns: dict[str, ColumnChunk]   # by leaf path, e.g. "final_means.list.element"
    n_row_groups: int


def _leaf_levels(schema: list[dict]) -> dict[str, tuple[int, int]]:
    """Walk the flat schema list, returning ``{leaf_path: (max_def, max_rep)}``."""
    levels: dict[str, tuple[int, int]] = {}
    idx = 0

    def walk(prefix: list[str], d: int, r: int) -> None:
        nonlocal idx
        el = schema[idx]
        idx += 1
        name = el[4].decode()
        rep = el.get(3, 0)          # 0 REQUIRED, 1 OPTIONAL, 2 REPEATED
        nchild = el.get(5)
        path = prefix + [name]
        if rep == 1:
            d += 1
        elif rep == 2:
            d += 1
            r += 1
        if nchild:
            for _ in range(nchild):
                walk(path, d, r)
        else:
            levels[".".join(path)] = (d, r)

    root = schema[0]
    idx = 1
    for _ in range(root.get(5) or 0):
        walk([], 0, 0)
    return levels


def parse_footer(tail: bytes) -> FileMeta:
    """Parse a Parquet footer from a buffer whose last bytes are the file's last bytes."""
    if tail[-4:] != b"PAR1":
        raise ValueError("not a parquet file (bad trailing magic)")
    mlen = struct.unpack("<I", tail[-8:-4])[0]
    if mlen + 8 > len(tail):
        raise ValueError(f"need {mlen + 8} trailing bytes, have {len(tail)}")
    md = read_struct(_TCompact(tail[-8 - mlen:-8]))
    levels = _leaf_levels(md[2])
    rgs = md[4]
    if len(rgs) != 1:
        raise ValueError(f"expected a single row group, found {len(rgs)}")
    cols: dict[str, ColumnChunk] = {}
    for cc in rgs[0][1]:
        cm = cc[3]
        path = ".".join(p.decode() for p in cm[3])
        dict_off = cm.get(11)
        data_off = cm[9]
        start = dict_off if dict_off is not None else data_off
        d, r = levels[path]
        cols[path] = ColumnChunk(
            path=path, physical=PHYS[cm[1]], codec=CODEC[cm[4]],
            num_values=cm[5], start=start, length=cm[7],
            dict_offset=dict_off, data_offset=data_off, max_def=d, max_rep=r,
        )
    return FileMeta(num_rows=md[3], created_by=md.get(6, b"").decode(),
                    columns=cols, n_row_groups=len(rgs))


def _decompress(codec: str, raw: bytes, uncompressed: int) -> bytes:
    if codec == "UNCOMPRESSED":
        return raw
    if codec == "SNAPPY":
        out = snappy_decompress(raw)
    elif codec == "GZIP":
        import gzip
        out = gzip.decompress(raw)
    else:
        raise NotImplementedError(f"codec {codec} is not supported here")
    if len(out) != uncompressed:
        raise ValueError(f"{codec}: got {len(out)} bytes, header said {uncompressed}")
    return out


def _plain(buf: bytes, physical: str, count: int) -> np.ndarray:
    dt = _NP.get(physical)
    if dt is None:
        raise NotImplementedError(f"PLAIN decode of {physical}")
    a = np.frombuffer(buf, dtype=dt, count=count)
    return a


def read_column(chunk_bytes: bytes, col: ColumnChunk) -> np.ndarray:
    """Decode one column chunk's values.  Requires all values non-null."""
    buf = chunk_bytes
    pos = 0
    dictionary: np.ndarray | None = None
    values: list[np.ndarray] = []
    seen = 0
    end = len(buf)
    lvl_w_def = int(col.max_def).bit_length()

    while pos < end and seen < col.num_values:
        r = _TCompact(buf, pos)
        ph = read_struct(r)
        hdr_end = r.p
        ptype = ph[1]
        usize, csize = ph[2], ph[3]
        page = _decompress(col.codec, buf[hdr_end:hdr_end + csize], usize)
        pos = hdr_end + csize

        if ptype == 2:  # DICTIONARY_PAGE
            dh = ph[7]
            dictionary = _plain(page, col.physical, dh[1])
            continue
        if ptype == 1:  # INDEX_PAGE
            continue
        if ptype == 3:
            raise NotImplementedError("DATA_PAGE_V2 is not supported here")
        if ptype != 0:
            raise ValueError(f"unknown page type {ptype}")

        dh = ph[5]
        nvals = dh[1]
        enc = ENC.get(dh[2], dh[2])
        if ENC.get(dh[3]) != "RLE" or ENC.get(dh[4]) != "RLE":
            raise NotImplementedError("only RLE-encoded levels are supported")

        q = 0
        if col.max_rep > 0:
            ln = struct.unpack_from("<i", page, q)[0]
            q += 4 + ln          # repetition levels: shape is known, skip
        n_present = nvals
        if col.max_def > 0:
            ln = struct.unpack_from("<i", page, q)[0]
            defs = rle_hybrid(page[q + 4:q + 4 + ln], lvl_w_def, nvals)
            q += 4 + ln
            n_present = int(np.count_nonzero(defs == col.max_def))
            if n_present != nvals:
                raise ValueError(
                    f"{col.path}: page has {nvals - n_present} nulls; this "
                    "reader assumes a fully populated column")
        body = page[q:]

        if enc == "PLAIN":
            values.append(_plain(body, col.physical, n_present).copy())
        elif enc in ("RLE_DICTIONARY", "PLAIN_DICTIONARY"):
            if dictionary is None:
                raise ValueError(f"{col.path}: dictionary-encoded page with no dictionary")
            w = body[0]
            idx = rle_hybrid(body[1:], w, n_present)
            values.append(dictionary[idx])
        else:
            raise NotImplementedError(f"{col.path}: page encoding {enc}")
        seen += nvals

    if seen != col.num_values:
        raise ValueError(f"{col.path}: decoded {seen} of {col.num_values} values")
    return np.concatenate(values) if len(values) > 1 else values[0]


# ---------------------------------------------------------------------------
# HTTP range fetching
# ---------------------------------------------------------------------------
class RangeFetcher:
    """GET byte ranges from a URL, retrying on the gateway's intermittent 5xx."""

    def __init__(self, url: str, *, retries: int = 6, timeout: float = 180.0,
                 verbose: bool = True) -> None:
        self.url = url
        self.retries = retries
        self.timeout = timeout
        self.verbose = verbose
        self.bytes_fetched = 0

    def get(self, start: int, length: int) -> bytes:
        """Fetch ``[start, start+length)``.  ``start < 0`` means a suffix range."""
        rng = f"bytes={start}-{start + length - 1}" if start >= 0 else f"bytes=-{length}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={"Range": rng, "User-Agent": "whestfloor-parquet-lite/1.0",
                             "Accept-Encoding": "identity"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status not in (200, 206):
                        raise urllib.error.HTTPError(
                            self.url, resp.status, "unexpected status", resp.headers, None)
                    data = resp.read()
                if resp.status == 200 and length < 1 << 30:
                    raise OSError("server ignored Range and sent the whole object")
                if len(data) != length:
                    raise OSError(f"range {rng}: got {len(data)} bytes, wanted {length}")
                self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # noqa: BLE001 - retry anything transient
                last = exc
                wait = min(2.0 * (2 ** attempt), 30.0)
                if self.verbose:
                    print(f"    [retry {attempt + 1}/{self.retries}] {rng}: "
                          f"{type(exc).__name__}: {exc}; sleeping {wait:.0f}s", flush=True)
                if attempt + 1 < self.retries:
                    time.sleep(wait)
        raise RuntimeError(f"range {rng} failed after {self.retries} attempts") from last


def read_columns_over_http(url: str, paths: list[str], *,
                           footer_bytes: int = 1 << 16,
                           verbose: bool = True) -> tuple[FileMeta, dict[str, np.ndarray]]:
    """Fetch and decode only ``paths`` from a remote Parquet file.

    Coalesces adjacent column chunks into single ranged GETs.
    """
    f = RangeFetcher(url, verbose=verbose)
    meta = parse_footer(f.get(-1, footer_bytes))
    missing = [p for p in paths if p not in meta.columns]
    if missing:
        raise KeyError(f"columns not in file: {missing}; have {sorted(meta.columns)}")

    chosen = sorted((meta.columns[p] for p in paths), key=lambda c: c.start)
    # coalesce runs that touch
    spans: list[list[int]] = []
    for c in chosen:
        if spans and c.start == spans[-1][1]:
            spans[-1][1] = c.start + c.length
        else:
            spans.append([c.start, c.start + c.length])
    blobs = {}
    for lo, hi in spans:
        if verbose:
            print(f"    GET bytes {lo:,}-{hi - 1:,}  ({hi - lo:,} B)", flush=True)
        blobs[lo] = f.get(lo, hi - lo)

    out: dict[str, np.ndarray] = {}
    for c in chosen:
        base = max(k for k in blobs if k <= c.start)
        off = c.start - base
        out[c.path] = read_column(blobs[base][off:off + c.length], c)
    if verbose:
        print(f"    total fetched: {f.bytes_fetched / 1e6:.2f} MB", flush=True)
    return meta, out
