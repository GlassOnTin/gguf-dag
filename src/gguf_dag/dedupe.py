"""FIDEDUPERANGE deduplication driven by gguf-dag manifests.

duperemove hashes a fixed grid from each file's offset 0, so it only finds
sharing when two files store identical bytes at offsets congruent mod the
block size.  gguf-dag's chunk digests are anchored at *tensor boundaries*,
which sit at different offsets in shard vs merged files, so the measured
cross-file sharing (98.3% between shard groups, 21.9% between quants) is
invisible to any grid hash.

This module closes that gap: a manifest alone reconstructs every chunk's
(file offset, length, digest); matching digests across two files give the
content-anchored overlap map, which is merged into long runs, clipped to the
4 KiB alignment FIDEDUPERANGE requires, and applied with the ioctl.  The
kernel compares contents before it shares an extent, so a mismatch (or a
live writer) can only ever cost us the dedup, not the data.
"""

import errno
import hashlib
import fcntl
import os
import struct

FIDEDUPERANGE = (3 << 30) | (24 << 16) | (0x94 << 8) | 54
_R_HDR = struct.Struct("<QQH2xI")   # src_offset, src_length, dest_count, reserved
_R_INFO = struct.Struct("<qQQii")   # dest_fd, dest_offset, bytes_deduped, status, reserved

FILE_DEDUPE_RANGE_SAME = 1      # already shared between these two files
FILE_DEDUPE_RANGE_DIFFERS = 2   # kernel verified different contents


def manifest_file_chunks(store, man):
    """(file_offset, length, digest) per chunk, in file order.

    Mirrors Store.add's tiling: each tensor's [start, start + n_bytes) span is
    tiled with min(chunk_size, remaining) pieces.
    """
    if man.get("chunk_size") != store.chunk_size:
        raise ValueError(
            f"manifest {man['id']} chunk_size {man.get('chunk_size')} != store {store.chunk_size}")
    mdir = store.manifest_path(man["id"]).parent
    refs = (mdir / "refs.bin").read_bytes()
    out = []
    for t in man["tensors"]:
        pos, end = t["start"], t["start"] + t["n_bytes"]
        base = t["refs_off"]
        for k in range(t["n_chunks"]):
            ln = min(store.chunk_size, end - pos)
            d = refs[base + 16 * k: base + 16 * (k + 1)]
            out.append((pos, ln, d))
            pos += ln
    return out


def bind_file(store, path):
    """Map a GGUF path to its manifest by header-region identity (and size)."""
    size = os.path.getsize(path)
    for man in store.list_manifests():
        if man["source"]["size"] != size:
            continue
        n = man["header"]["raw_bytes"]
        if n <= 0 or n > size:
            continue
        with open(path, "rb") as f:
            head = f.read(n)
        if hashlib.blake2b(head, digest_size=16).digest().hex() == man["id"]:
            return man
    return None


def find_shared_runs(chunks_a, chunks_b):
    """Merged content-anchored runs shared by two files: [off_a, off_b, size].

    Runs chain only where both files advance contiguously, so alignment
    padding between tensors (up to 32 B) naturally splits runs.
    """
    bmap = {}  # digest -> (off, ln); first occurrence wins
    for off, ln, d in chunks_b:
        if d not in bmap:
            bmap[d] = (off, ln)
    runs = []
    for off_a, ln_a, d in chunks_a:
        hit = bmap.get(d)
        if hit is not None:
            runs.append((off_a, ln_a, hit[0]))
    runs.sort()
    merged = []
    cur = None
    for off_a, ln, off_b in runs:
        if cur and off_a == cur[0] + cur[2] and off_b == cur[1] + cur[2]:
            cur[2] += ln
        else:
            if cur:
                merged.append(cur)
            cur = [off_a, off_b, ln]
    if cur:
        merged.append(cur)
    return merged


def clip_aligned(off_a, off_b, size, align=4096):
    """Clip a shared run to what FIDEDUPERANGE can actually dedupe.

    The ioctl needs both offsets block-aligned, and the two windows must map
    the *same content position*, so the usable window starts at a common
    shift t with off_a + t and off_b + t both aligned.  That only exists
    when the run's phases agree: off_a ≡ off_b (mod align).  Otherwise the
    run yields nothing and we report None rather than silently misaligning.
    """
    mask = align - 1
    if (off_a ^ off_b) & mask:
        return None
    s = (off_a + align - 1) & ~mask
    e = (off_a + size) & ~mask
    if e - s < align:
        return None
    return (s, off_b + (s - off_a), e - s)


def ranges_for_pair(store, man_a, man_b):
    """Aligned dedupe ranges [(off_a, off_b, len), ...] for two manifests."""
    ca = manifest_file_chunks(store, man_a)
    cb = manifest_file_chunks(store, man_b)
    out = []
    for off_a, off_b, size in find_shared_runs(ca, cb):
        r = clip_aligned(off_a, off_b, size)
        if r is not None:
            out.append(r)
    return out


def fideduperange(fd_src, off_src, fd_dst, off_dst, length):
    """One FIDEDUPERANGE call with a single dest; loops on partial results.

    Returns (total_mapped, note) where note is None or a short reason the
    loop stopped early.
    """
    left = length
    total = 0
    note = None
    s = off_src
    d = off_dst
    while left > 0:
        buf = _R_HDR.pack(s, left, 1, 0) + _R_INFO.pack(fd_dst, d, 0, 0, 0)
        try:
            r = fcntl.ioctl(fd_src, FIDEDUPERANGE, bytes(buf))
        except OSError as e:
            return total, f"ioctl {-e.errno} {os.strerror(e.errno)}"
        _, _, mapped, status, _ = _R_INFO.unpack_from(r, 24)
        if status < 0:
            note = f"status {-status} {os.strerror(-status)}"
            break
        if status == FILE_DEDUPE_RANGE_DIFFERS:
            note = "DIFFERS"
            break
        if status == FILE_DEDUPE_RANGE_SAME:
            # range already shares an extent between these files
            total += left
            break
        if mapped == 0:
            note = "no progress (0 bytes)"
            break
        total += mapped
        s += mapped
        d += mapped
        left -= mapped
    return total, note