"""Content-addressed store for GGUF files.

Layout under the store root:
  chunks.pack     append-only concatenation of unique 64 KiB (configurable)
                  chunks, first-seen order
  chunks.idx      flat sorted records: blake2b-16 digest, u64 pack offset,
                  u32 chunk byte length, 4 bytes pad = 32 B/record; bisect
                  lookups over an mmap
  manifests/<id>/ manifest.json + refs.bin (per-tensor chunk-digest streams)
                  + header.bin (zlib-compressed header region)
  lock            flock, single writer

manifest.json is small and human-readable; the heavy per-tensor chunk digest
lists live in the binary refs.bin so a 100 GB model needs only a few-KB
manifest.  Rebuild = header blob + per-tensor chunk streams, byte-identical
to the source by construction.

Optional HMAC-BLAKE2b signing authenticates the manifest; the hash chain
(chunk digests) detects payload corruption.
"""

import fcntl
import hashlib
import hmac
import json
import mmap
import os
import struct
import zlib
from pathlib import Path

from .gguf import GGUFFile, GGUFError

IDX_REC = 32
MANIFEST_VERSION = 1


def _blake(data, size=16):
    return hashlib.blake2b(data, digest_size=size).digest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class Store:
    def __init__(self, root, chunk_size=64 * 1024):
        self.root = Path(root)
        (self.root / "manifests").mkdir(parents=True, exist_ok=True)
        if not (1 << 12) <= chunk_size <= (1 << 26):
            raise GGUFError(f"chunk_size {chunk_size} outside [4KiB, 64MiB]")
        self.chunk_size = int(chunk_size)
        self._lock_fd = open(self.root / "lock", "w")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        self.pack_fp = open(self.root / "chunks.pack", "a+b")
        self.pack_size = os.fstat(self.pack_fp.fileno()).st_size
        self._idx_fp = open(self.root / "chunks.idx", "a+b")
        self._idx_size = os.fstat(self._idx_fp.fileno()).st_size
        if self._idx_size % IDX_REC:
            raise GGUFError(f"corrupt index size {self._idx_size}")
        self._idx_mm = None
        self._remap_idx()
        self.idx_count = self._idx_size // IDX_REC
        self._pending = {}  # digest -> (pack_offset, byte_size); queued this session
        self._hmac_key = None
        self.open = True

    # -- index ---------------------------------------------------------
    def _remap_idx(self):
        if self._idx_mm is not None:
            self._idx_mm.close()
            self._idx_mm = None
        if self._idx_size:
            self._idx_mm = mmap.mmap(self._idx_fp.fileno(), 0, access=mmap.ACCESS_READ)

    def _idx_get(self, digest):
        """Return (pack_offset, size) or None. Checks queued pending first,
        then the stable sorted index."""
        hit = self._pending.get(digest)
        if hit is not None:
            return hit
        lo, hi = 0, self.idx_count
        while lo < hi and self._idx_mm is not None:
            mid = (lo + hi) >> 1
            o = mid * IDX_REC
            key = self._idx_mm[o:o + 16]
            if key == digest:
                off, ln = struct.unpack_from("<QI", self._idx_mm, o + 16)
                return off, ln
            if key < digest:
                lo = mid + 1
            else:
                hi = mid
        return None

    def _flush_idx(self):
        """Merge pending entries into the sorted index file (streaming merge)."""
        if not self._pending:
            return
        pend = sorted(self._pending.items())
        tmp_path = self.root / "chunks.idx.tmp"
        with open(tmp_path, "wb") as tmp:
            i = pi = 0
            while i < self.idx_count or pi < len(pend):
                if pi >= len(pend) or (i < self.idx_count and
                                       self._idx_mm[i * IDX_REC: i * IDX_REC + 16] <= pend[pi][0]):
                    o = i * IDX_REC
                    tmp.write(self._idx_mm[o:o + IDX_REC])
                    i += 1
                else:
                    d, (off, ln) = pend[pi]
                    tmp.write(struct.pack("<16sQI4x", d, off, ln))
                    pi += 1
        os.replace(tmp_path, self.root / "chunks.idx")
        self._pending.clear()
        self._idx_fp.close()
        self._idx_fp = open(self.root / "chunks.idx", "a+b")
        self._idx_size = os.path.getsize(self.root / "chunks.idx")
        self.idx_count = self._idx_size // IDX_REC
        self._remap_idx()

    # -- add -----------------------------------------------------------
    def add(self, path, hmac_key=None):
        """Split one GGUF into the store. Returns the manifest id."""
        gf = GGUFFile(path)
        with open(path, "rb") as f:
            src = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                refs = bytearray()
                tensors = []
                new_chunks = new_bytes = total_chunks = 0
                for t in gf.tensors:
                    refs_off = len(refs)
                    n_chunks = 0
                    pos, end = t["start"], t["start"] + t["n_bytes"]
                    while pos < end:
                        ln = min(self.chunk_size, end - pos)
                        c = src[pos:pos + ln]
                        d = _blake(c)
                        if self._idx_get(d) is None:
                            # append to pack; sequential writes, no seek needed
                            self.pack_fp.write(c)
                            self._pending[d] = (self.pack_size, ln)
                            self.pack_size += ln
                            new_chunks += 1
                            new_bytes += ln
                        refs += d
                        n_chunks += 1
                        total_chunks += 1
                        pos += ln
                    tensors.append({
                        "name": t["name"], "ggml_type": t["ggml_type"],
                        "dims": t["dims"], "start": t["start"],
                        "n_bytes": t["n_bytes"], "n_chunks": n_chunks,
                        "refs_off": refs_off,
                    })
            finally:
                src.close()

        self.pack_fp.flush()
        os.fsync(self.pack_fp.fileno())

        mid = _blake(gf.header_bytes, 16).hex()
        mdir = self.root / "manifests" / mid
        mdir.mkdir(exist_ok=True)
        (mdir / "refs.bin").write_bytes(bytes(refs))
        hdr_blob = zlib.compress(gf.header_bytes, 6)
        (mdir / "header.bin").write_bytes(hdr_blob)
        manifest = {
            "format": "gguf-dag",
            "version": MANIFEST_VERSION,
            "id": mid,
            "chunk_size": self.chunk_size,
            "source": {"name": os.path.basename(path), "size": gf.file_size},
            "header": {
                "raw_bytes": len(gf.header_bytes),
                "compressed_bytes": len(hdr_blob),
                "blake2b_256": _blake(gf.header_bytes, 32).hex(),
                "blob": "header.bin",
            },
            "tensors": tensors,
            "refs": {
                "blob": "refs.bin",
                "bytes": len(refs),
                "blake2b_256": _blake(bytes(refs), 32).hex(),
            },
            "chunks": {
                "count": total_chunks,
                "new_unique": new_chunks,
                "new_unique_bytes": new_bytes,
            },
        }
        if hmac_key is not None:
            manifest["hmac_sha"] = "blake2b"
            payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            tag = hmac.new(hmac_key, payload, digestmod=lambda *a, **k: hashlib.blake2b(*a, digest_size=32, **k)).hexdigest()
            manifest["hmac"] = tag
        (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        self._flush_idx()
        return mid

    # -- read ----------------------------------------------------------
    def manifest_path(self, ref):
        """Resolve an id prefix (or exact id) to a manifest.json path."""
        mroot = self.root / "manifests"
        if (mroot / ref / "manifest.json").exists():
            return mroot / ref / "manifest.json"
        matches = sorted(p.name for p in mroot.iterdir() if p.name.startswith(ref))
        if len(matches) == 1:
            return mroot / matches[0] / "manifest.json"
        if not matches:
            raise GGUFError(f"no manifest matching '{ref}'")
        raise GGUFError(f"'{ref}' ambiguous: {matches[:6]}{'...' if len(matches) > 6 else ''}")

    def load_manifest(self, ref):
        return json.loads(self.manifest_path(ref).read_text())

    def rebuild(self, ref, out_path):
        """Rebuild a byte-identical GGUF. Returns sha256 hex of the output."""
        man = self.load_manifest(ref)
        mdir = self.manifest_path(ref).parent
        if self._pending:
            raise GGUFError("add incomplete")
        pack = mmap.mmap(self.pack_fp.fileno(), 0, access=mmap.ACCESS_READ) if self.pack_size else None
        refs_path = mdir / "refs.bin"
        refs = None
        if os.path.getsize(refs_path):
            fd = os.open(refs_path, os.O_RDONLY)
            refs = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)
        sha = hashlib.sha256()
        try:
            blob = (mdir / "header.bin").read_bytes()
            if len(blob) != man["header"]["compressed_bytes"]:
                raise GGUFError("header blob size mismatch")
            header = zlib.decompress(blob)
            if _blake(header, 32).hex() != man["header"]["blake2b_256"]:
                raise GGUFError("header blob corrupted")
            sha.update(header)
            with open(out_path, "wb") as out:
                out.write(header)
                for t in man["tensors"]:
                    n = t["n_chunks"]
                    stream = refs[t["refs_off"]: t["refs_off"] + 16 * n]
                    remaining = t["n_bytes"]
                    for k in range(n):
                        d = bytes(stream[k * 16:k * 16 + 16])
                        hit = self._idx_get(d)
                        if hit is None:
                            raise GGUFError(f"chunk {d.hex()} missing from index")
                        off, ln = hit
                        take = min(ln, remaining)
                        piece = pack[off:off + take]
                        sha.update(piece)
                        out.write(piece)
                        remaining -= take
                    if remaining:
                        raise GGUFError(f"tensor '{t['name']}' rebuilt short by {remaining} bytes")
            if os.path.getsize(out_path) != man["source"]["size"]:
                raise GGUFError("rebuilt file size differs from manifest")
        finally:
            if pack is not None:
                pack.close()
            if refs is not None:
                refs.close()
        return sha.hexdigest()

    def verify(self, ref, hmac_key=None):
        """Recompute the hash chain for one manifest.

        Returns (n_chunks_checked, list_of_problems) — empty list means the
        manifest and all referenced chunks are intact.
        """
        man = self.load_manifest(ref)
        probs = []
        mdir = self.manifest_path(ref).parent
        if hmac_key is not None:
            if "hmac" not in man:
                probs.append("manifest has no HMAC but a key was supplied")
            else:
                body = {k: v for k, v in man.items() if k != "hmac"}
                payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                want = hmac.new(hmac_key, payload,
                                digestmod=lambda *a, **k: hashlib.blake2b(*a, digest_size=32, **k)).hexdigest()
                if not hmac.compare_digest(want, man["hmac"]):
                    probs.append("manifest HMAC mismatch")
        blob = (mdir / "header.bin").read_bytes()
        if len(blob) != man["header"]["compressed_bytes"]:
            probs.append("header blob compressed size mismatch")
        elif _blake(zlib.decompress(blob), 32).hex() != man["header"]["blake2b_256"]:
            probs.append("header blob hash mismatch")
        refs_bytes = (mdir / "refs.bin").read_bytes()
        if len(refs_bytes) != man["refs"]["bytes"]:
            probs.append("refs blob size mismatch")
        elif _blake(refs_bytes, 32).hex() != man["refs"]["blake2b_256"]:
            probs.append("refs blob hash mismatch")
        if probs:
            return 0, probs
        pack = mmap.mmap(self.pack_fp.fileno(), 0, access=mmap.ACCESS_READ) if self.pack_size else None
        n_checked = 0
        try:
            for t in man["tensors"]:
                remaining = t["n_bytes"]
                stream = refs_bytes[t["refs_off"]: t["refs_off"] + 16 * t["n_chunks"]]
                for k in range(t["n_chunks"]):
                    d = stream[k * 16:k * 16 + 16]
                    hit = self._idx_get(d)
                    if hit is None:
                        probs.append(f"tensor '{t['name']}' refs chunk {d.hex()} absent")
                        continue
                    off, ln = hit
                    take = min(ln, remaining)
                    if _blake(pack[off:off + take], 16) != d:
                        probs.append(f"tensor '{t['name']}' chunk {d.hex()} payload hash mismatch")
                    n_checked += 1
                    remaining -= take
                if remaining:
                    probs.append(f"tensor '{t['name']}' spans {remaining} bytes beyond its chunk list")
        finally:
            if pack is not None:
                pack.close()
        return n_checked, probs

    # -- introspection ---------------------------------------------------
    def list_manifests(self):
        out = []
        for d in sorted((self.root / "manifests").iterdir()):
            mj = d / "manifest.json"
            if mj.exists():
                man = json.loads(mj.read_text())
                out.append(man)
        return out

    def stats(self):
        mans = self.list_manifests()
        total_src = sum(m["source"]["size"] for m in mans)
        total_chunks = sum(m["chunks"]["count"] for m in mans)
        new_bytes = sum(m["chunks"]["new_unique_bytes"] for m in mans)
        index_bytes = os.path.getsize(self.root / "chunks.idx")
        per_manifest = {m["id"]: m["source"]["name"] for m in mans}
        overhead = index_bytes / max(self.pack_size, 1)
        return {
            "n_files": len(mans),
            "total_source_bytes": total_src,
            "pack_bytes": self.pack_size,
            "dedup_saving": 1 - self.pack_size / total_src if total_src else 0.0,
            "chunk_records": self.idx_count,
            "overhead_index_pct": overhead * 100,
            "total_chunks": total_chunks,
            "new_unique_bytes_sum": new_bytes,
            "labels": per_manifest,
        }

    def close(self):
        if not self.open:
            return
        self._flush_idx()
        self.pack_fp.close()
        self._idx_fp.close()
        if self._idx_mm is not None:
            self._idx_mm.close()
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        self._lock_fd.close()
        self.open = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()