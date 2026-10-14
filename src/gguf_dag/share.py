"""Cross-process weight-pool tooling for the llama.cpp fork's --share-ipc.

The fork vendored as deps/llama.cpp (GlassOnTin/llama.cpp, branch
ipc-share-vmm) shares CUDA weight tensors between processes through VMM
segments that llama-share-registry pins under (sha256 tensor digest, PCI bus
id).  llama-server instances publish while loading; llama-share-prewarm can
seed the pool from a manifest plus a GGUF with no llama process at all.

This module makes gguf-dag the source of the manifests:

  - a publish-run manifest written next to the GGUF records exactly the
    tensors the loader reads (for tied-embedding models like qwen2.5 it
    reads output.weight and never token_embd.weight; nothing structural in
    the file says which applies), so it is used whenever its size+mtime gate
    still matches the file;
  - otherwise digests are computed here, per tensor, from the store chunk
    stream (blake2b-verified on each read) or straight from the GGUF file,
    and rendered in the same TSV format.  Store-derived digests are cached
    under the store manifest id, which is content-keyed, so they survive
    mtime churn and are only ever computed once.

Registry introspection speaks the wire format from src/llama-share.h: a
packed 64-byte little-endian header, one message per connection, with the
bus id trailing the query.
"""

import hashlib
import os
import re
import shutil
import socket
import struct
import subprocess
from pathlib import Path

from . import ggml
from .gguf import GGUFFile, GGUFError
from .store import Store
from .dedupe import bind_file

_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")

MAGIC = 0x50485347  # "GSHP"
VERSION = 1
QUERY, REGISTER = 1, 2
ST_OK, ST_MISS, ST_HIT, ST_ERR = 0, 1, 2, 3

_HDR = struct.Struct("<6IQ32s")  # magic, version, type, status, name_len, bus_len, seg_size, digest

MAX_PAD = 31  # GGUF pads tensor payloads to a 32-byte alignment at most


class ShareError(ValueError):
    pass


# --- registry wire protocol ------------------------------------------

def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        piece = sock.recv(n - len(buf))
        if not piece:
            raise ShareError("registry closed the connection mid-message")
        buf.extend(piece)
    return bytes(buf)


class RegistryClient:
    """One-message-per-connection registry client (protocol invariant)."""

    def __init__(self, path, timeout=2.0):
        self.path = os.fspath(path)
        self.timeout = timeout

    def query(self, digest, bus=None):
        """Ask the registry about one digest (32 raw bytes), filtered by bus id.

        Returns ("hit", seg_size, bus) or ("miss", None, None); raises
        ShareError on transport failures and protocol-level errors.
        """
        if not isinstance(digest, (bytes, bytearray)) or len(digest) != 32:
            raise ShareError("query digest must be 32 raw bytes")
        bus_len = len(bus) + 1 if bus is not None else 0
        trail = bus.encode() + b"\0" if bus_len else b""
        req = _HDR.pack(MAGIC, VERSION, QUERY, ST_OK, 0, bus_len, 0, bytes(digest)) + trail
        hit_bus = None
        try:
            # the whole reply (header, plus the bus trail on HIT) must be read
            # before the with-block closes the socket
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect(self.path)
                s.sendall(req)
                (magic, version, _rtype, status, _nl,
                 rbus_len, seg_size, _d) = _HDR.unpack(_recv_exact(s, _HDR.size))
                if magic != MAGIC:
                    raise ShareError(f"registry reply has bad magic {magic:#x}")
                if version != VERSION:
                    raise ShareError(f"registry reply version {version} != {VERSION}")
                if status == ST_HIT:
                    bus_trail = _recv_exact(s, rbus_len) if rbus_len else b""
                    hit_bus = bus_trail[:-1].decode("utf-8", "replace") if rbus_len else None
        except OSError as e:
            raise ShareError(f"cannot reach registry at {self.path}: {e}") from e
        if status == ST_MISS:
            return ("miss", None, None)
        if status == ST_ERR:
            raise ShareError("registry reported an error on query")
        if status == ST_HIT:
            return ("hit", seg_size, hit_bus)
        raise ShareError(f"registry reply status {status} on query")


# --- share manifests --------------------------------------------------

def parse_share_manifest(path):
    """Parse a publish-format manifest. Returns (total, max_mtime, entries).

    Entries are (name, digest_hex, nbytes) in file order.  The format is the
    one llama-share writes: a '#gguf-share-manifest\\t<total>\\t<max_mtime>'
    first line (total = sum of weight-file sizes, max_mtime = max of their
    mtimes), then 'name\\t<64-hex sha256>\\t<nbytes>' lines.
    """
    total = mtime = None
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("#gguf-share-manifest\t"):
                    parts = line.split("\t")
                    if len(parts) != 3 or total is not None:
                        raise ShareError(f"{path}: malformed manifest header")
                    total, mtime = parts[1], parts[2]
                    if not (total.isdigit() and mtime.isdigit()):
                        raise ShareError(f"{path}: malformed manifest header")
                    total, mtime = int(total), int(mtime)
                continue
            fields = line.split("\t")
            digest_ok = (len(fields) == 3 and len(fields[1]) == 64
                         and all(c in "0123456789abcdef" for c in fields[1])
                         and fields[2].isdigit() and int(fields[2], 10) >= 1)
            if not digest_ok:
                raise ShareError(f"{path}: malformed manifest entry '{line[:64]}'")
            entries.append((fields[0], fields[1], int(fields[2], 10)))
    if total is None:
        raise ShareError(f"{path}: missing '#gguf-share-manifest' header")
    return total, mtime, entries


def render_share_manifest(total, max_mtime, entries):
    lines = [f"#gguf-share-manifest\t{total}\t{max_mtime}"]
    lines += [f"{name}\t{digest}\t{nbytes}" for name, digest, nbytes in entries]
    return "".join(line + "\n" for line in lines)


def file_state(paths):
    """(sum of file sizes, max mtime) — the values the manifest header gates on."""
    total = mtime = 0
    for p in paths:
        st = os.stat(p)
        total += st.st_size
        mtime = max(mtime, int(st.st_mtime))
    return total, mtime


def expand_shards(path):
    """Expand a split GGUF path ('...-00001-of-0000N.gguf') into all N shard paths.

    Matches llama.cpp's loader: when every shard 1..N exists on disk, the model
    spans the ordered list and the manifest sits next to shard 1
    (<shard1>.share.tsv), gated on the sum of all shard sizes and max of all
    shard mtimes.  Single-file GGUFs return [os.fspath(path)].
    """
    s = os.fspath(path)
    m = _SHARD_RE.match(s)
    if m:
        prefix, idx_s, count_s = m.groups()
        idx, count = int(idx_s, 10), int(count_s, 10)
        if 1 <= idx <= count and count > 1:
            shards = [f"{prefix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
            if all(os.path.exists(p) for p in shards):
                return shards
    return [s]


def default_manifest(path):
    return Path(expand_shards(path)[0] + ".share.tsv")


def fresh_publish_manifest(gguf_path, mpath=None):
    """Use the publish-run manifest next to the GGUF if its gate still matches."""
    shards = expand_shards(gguf_path)
    mpath = Path(mpath) if mpath else Path(shards[0] + ".share.tsv")
    if not mpath.exists():
        return None
    total, mtime, _entries = parse_share_manifest(mpath)
    if (total, mtime) == file_state(shards):
        return mpath
    return None


def _write_atomic(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _digests_from_store(store, man):
    """Per-tensor sha256 digests rebuilt from the store's chunk stream.

    Each span chunk is blake2b-verified as it is read, so a corrupted pack or
    index fails loudly.  The payload occupies a strict prefix of the span;
    a span/payload mismatch beyond GGUF's max padding means the type table is
    wrong, which is exactly what we want the check to catch.
    """
    mdir = store.manifest_path(man["id"]).parent
    refs = (mdir / "refs.bin").read_bytes()
    pack_fd = store.pack_fp.fileno()
    out = []
    for t in man["tensors"]:
        payload = ggml.payload_nbytes(t["dims"], t["ggml_type"])
        if not 0 <= t["n_bytes"] - payload <= MAX_PAD:
            raise ShareError(
                f"tensor '{t['name']}': span {t['n_bytes']} vs payload {payload} "
                f"({ggml.type_name(t['ggml_type'])}) differs by more than {MAX_PAD} "
                "pad bytes; the ggml size table disagrees with this file")
        sha = hashlib.sha256()
        remaining = payload
        base = t["refs_off"]
        for k in range(t["n_chunks"]):
            d = refs[base + 16 * k: base + 16 * (k + 1)]
            hit = store._idx_get(d)
            if hit is None:
                raise ShareError(f"tensor '{t['name']}' chunk {d.hex()} missing from index")
            off, ln = hit
            # read and blake2b-verify the FULL pack chunk (its tail may be the
            # span's inter-tensor padding, which is not part of the payload)
            chunk = os.pread(pack_fd, ln, off)
            if len(chunk) != ln:
                raise ShareError(f"tensor '{t['name']}' chunk {d.hex()} short read from pack")
            if hashlib.blake2b(chunk, digest_size=16).digest() != d:
                raise ShareError(f"tensor '{t['name']}' chunk {d.hex()} payload hash mismatch")
            take = min(ln, remaining)
            sha.update(chunk[:take])
            remaining -= take
        if remaining:
            raise ShareError(f"tensor '{t['name']}' chunk stream short by {remaining} bytes")
        out.append((t["name"], sha.hexdigest(), payload))
    return out


def _digests_from_file(path):
    """Per-tensor sha256 digests read straight from the GGUF payload bytes."""
    gf = GGUFFile(path)
    out = []
    with open(path, "rb") as f:
        for t in gf.tensors:
            payload = ggml.payload_nbytes(t["dims"], t["ggml_type"])
            if not 0 <= t["n_bytes"] - payload <= MAX_PAD:
                raise ShareError(
                    f"tensor '{t['name']}': span {t['n_bytes']} vs payload {payload} "
                    f"({ggml.type_name(t['ggml_type'])}) differs by more than {MAX_PAD} "
                    "pad bytes; the ggml size table disagrees with this file")
            sha = hashlib.sha256()
            f.seek(t["start"])
            left = payload
            while left:
                piece = f.read(min(left, 1 << 23))
                if not piece:
                    raise ShareError(f"tensor '{t['name']}': unexpected EOF")
                sha.update(piece)
                left -= len(piece)
            out.append((t["name"], sha.hexdigest(), payload))
    return out


def store_tensor_digests(store, man):
    """Digests for one in-store GGUF, cached under the store manifest id.

    The cache is content-keyed and write-once (store manifest ids change
    whenever the header bytes change, which is the only thing that could
    invalidate digests beyond payload tampering, and tampered payloads fail
    the blake2b check on read).  Existing cache files are trusted.
    """
    cache = store.manifest_path(man["id"]).parent / "tensor-digests.tsv"
    if cache.exists():
        entries = []
        for raw in cache.read_text(encoding="utf-8").splitlines():
            if raw.startswith("#") or not raw:
                continue
            name, digest, nbytes = raw.split("\t")
            entries.append((name, digest, int(nbytes, 10)))
        return entries
    entries = _digests_from_store(store, man)
    body = "".join(f"{n}\t{d}\t{b}\n" for n, d, b in entries)
    _write_atomic(cache, body)
    return entries


def ensure_manifest(gguf_path, store_root=None, out=None, force=False):
    """Produce a share manifest for one GGUF (or shard group). Returns (path, entries, source).

    Preference order: a fresh publish-run manifest next to the file (exact
    loaded-tensor record); the store's chunk stream; the GGUF file(s) directly.
    All fresh generations render to <shard1>.share.tsv unless out is given,
    because that is the path llama-server's --share-ipc reads by default.

    force=True permits overwriting a manifest next to the file that is NOT a
    fresh publish-run record (i.e. one whose gate no longer matches, meaning
    the file changed after whatever produced it).
    """
    shards = expand_shards(gguf_path)
    mpath = Path(out) if out else Path(shards[0] + ".share.tsv")

    fresh = fresh_publish_manifest(gguf_path) if out is None else None
    if fresh is not None and fresh == mpath:
        _total, _mtime, entries = parse_share_manifest(fresh)
        return fresh, entries, "publish-run manifest (fresh gate)"

    entries = None
    if store_root is not None:
        with Store(store_root) as st:
            mans = [bind_file(st, s) for s in shards]
            if all(m is not None for m in mans):
                entries = []
                for m in mans:
                    entries.extend(store_tensor_digests(st, m))
                ids = ",".join(m["id"][:12] for m in mans)
                source = f"store manifest{'s' if len(mans) > 1 else ''} {ids} "
                source += f"({len(entries)} tensors incl. any the loader may skip)"
    if entries is None:
        entries = []
        for s in shards:
            entries.extend(_digests_from_file(s))
        source = "GGUF file direct" if len(shards) == 1 else f"GGUF {len(shards)} shards direct"

    if mpath.exists() and force is False:
        raise ShareError(
            f"{mpath} already exists and is not a fresh publish-run record; "
            "the model file changed after it was written. Rerun with --force to "
            "overwrite, or delete it.")
    total, mtime = file_state(shards)
    _write_atomic(mpath, render_share_manifest(total, mtime, entries))
    return mpath, entries, source


# --- registry queries -------------------------------------------------

def normalize_bus(bus):
    """'00000000:49:00.0' (nvidia-smi) -> '0000:49:00.0' (cudaDeviceGetPCIBusId),
    i.e. the domain is written as %04x by CUDA and %08x by nvidia-smi."""
    bus = bus.strip().lower()
    if len(bus) == 16 and bus[8] == ":" and bus[:8].isdigit():
        bus = bus[4:]
    return bus


def probe_buses():
    """PCI bus ids of GPUs with the CUDA form, or [] when nvidia-smi is absent."""
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False)
    if out.returncode != 0:
        return []
    return [normalize_bus(b) for b in out.stdout.splitlines() if b.strip()]