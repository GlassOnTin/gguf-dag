"""Tests for the manifest-driven FIDEDUPERANGE dedupe.

Pure-planning tests (chunk reconstruction, run merging, 4 KiB clipping) run
anywhere.  The ioctl round-trip is fs-gated: it probes FIDEDUPERANGE on
scratch copies first and skips on filesystems without support (tmpfs, ext4).
"""

import hashlib
import fcntl
import os
import shutil
import struct
import tempfile
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_dag.dedupe import (bind_file, clip_aligned, find_shared_runs,
                             fideduperange, manifest_file_chunks,
                             ranges_for_pair)
from gguf_dag.store import Store

from tests.util import build_gguf

BLOCK = 4096

FS_IOC_FIEMAP = (3 << 30) | (32 << 16) | (0x66 << 8) | 11
_FIEMAP_HDR = struct.Struct("<QQIIII")
_FIEMAP_EXT = struct.Struct("<QQQQQI3I")
FIEMAP_EXTENT_SHARED = 0x00002000


def physical_extents(fd, start, length):
    """[(logical, physical, length, flags)] covering [start, start+length).

    FIEMAP may return whole extents reaching past the range, so each is
    clipped back before returning.
    """
    out = []
    pos = start
    while pos < start + length:
        n = 64
        # mutable buffer: fcntl.ioctl rejects bytes longer than _IOC_SIZE
        buf = bytearray(n * _FIEMAP_EXT.size + _FIEMAP_HDR.size)
        _FIEMAP_HDR.pack_into(buf, 0, pos, start + length - pos, 0, 0, n, 0)
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buf)
        _, _, _, mapped, _, _ = _FIEMAP_HDR.unpack_from(buf)
        if mapped == 0:
            break
        for i in range(mapped):
            lo, ph, ln, _, _, flags = _FIEMAP_EXT.unpack_from(
                buf, _FIEMAP_HDR.size + i * _FIEMAP_EXT.size)[:6]
            out.append((lo, ph, ln, flags))
        pos = out[-1][0] + out[-1][2]
    res = []
    for lo, ph, ln, flags in out:
        a, b = max(lo, start), min(lo + ln, start + length)
        if a < b:
            res.append((a, ph + (a - lo), b - a, flags))
    return res


def _rmtree(p):
    shutil.rmtree(p, ignore_errors=True)


def fs_supports_dedupe(test_dir):
    a = os.path.join(test_dir, ".fsprobe_a")
    b = os.path.join(test_dir, ".fsprobe_b")
    payload = os.urandom(BLOCK) * 4
    for p in (a, b):
        with open(p, "wb") as f:
            f.write(b"\x00" * BLOCK + payload)
    fa = os.open(a, os.O_RDONLY)
    fb = os.open(b, os.O_RDONLY)
    try:
        mapped, note = fideduperange(fa, BLOCK, fb, BLOCK, 4 * BLOCK)
        return mapped == 4 * BLOCK, note
    finally:
        os.close(fa)
        os.close(fb)
        os.remove(a)
        os.remove(b)


def scratch_dir(parent=None):
    """A test scratch dir on the real fs (tmpfs has no FIDEDUPERANGE)."""
    if parent is None:
        parent = os.environ.get("GGUF_DAG_IOCTL_DIR",
                                os.path.expanduser("~/.cache/gguf-dag-tests"))
    Path(parent).mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="dedupe-", dir=parent)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gguf-dag-dedupe-")
        self.st = Store(os.path.join(self.tmp, "store"))

    def tearDown(self):
        self.st.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mk(self, name, payloads):
        tensors = [(f"t{i}", [n // 64, 64], 0, p) for i, (n, p) in enumerate(payloads)]
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(build_gguf(tensors=tensors, alignment=32, inter_gap=0, tail_pad=0))
        return path

    def test_chunk_reconstruction_matches_add_tiling(self):
        c = self.st.chunk_size
        p = self._mk("a.gguf", [(2 * c, os.urandom(2 * c)),
                                (c + 1000, os.urandom(c + 1000))])
        man = self.st.load_manifest(self.st.add(p))
        chunks = manifest_file_chunks(self.st, man)
        with open(p, "rb") as f:
            for off, ln, d in chunks:
                f.seek(off)
                self.assertEqual(hashlib.blake2b(f.read(ln), digest_size=16).digest(), d)
        # offsets chain per tensor exactly as the store tiles spans
        seen = 0
        for t in man["tensors"]:
            pos = t["start"]
            for off, ln, d in chunks[seen:seen + t["n_chunks"]]:
                self.assertEqual(off, pos)
                pos += ln
            self.assertEqual(pos, t["start"] + t["n_bytes"])
            seen += t["n_chunks"]
        self.assertEqual(seen, len(chunks))

    def test_clip_aligned_congruent_and_mismatched(self):
        # Same content position on both sides must be block-aligned on each:
        # off_a and off_b congruent mod 4096 -> common aligned window; else None.
        s, b, ln = clip_aligned(8243, 8243 + BLOCK, 16384)
        self.assertEqual(s, 12288)
        self.assertEqual(b, 8243 + BLOCK + (12288 - 8243))
        self.assertEqual(b % BLOCK, 0)
        self.assertEqual(ln, (8243 + 16384) // BLOCK * BLOCK - 12288)
        self.assertIsNone(clip_aligned(8243, 8243 + BLOCK + 51, 16384))

    def test_shared_runs_at_congruent_offsets_clip_to_aligned_ranges(self):
        # Sharing happens at chunk granularity: identical tensor payloads
        # produce identical digests.  Both files use the same tensor names,
        # so their header regions are the same size and tensor starts differ
        # by exactly the pad difference -- kept a multiple of 4 KiB here.
        shared = os.urandom(3 * BLOCK)
        pa = self._mk("ra.gguf", [(2 * BLOCK, os.urandom(2 * BLOCK)),
                                  (len(shared), shared)])
        pb = self._mk("rb.gguf", [(1 * BLOCK, os.urandom(1 * BLOCK)),
                                  (len(shared), shared)])
        ma = self.st.load_manifest(self.st.add(pa))
        mb = self.st.load_manifest(self.st.add(pb))
        self.assertEqual((ma["tensors"][1]["start"] - mb["tensors"][1]["start"]) % BLOCK, 0)
        runs = find_shared_runs(manifest_file_chunks(self.st, ma),
                                manifest_file_chunks(self.st, mb))
        self.assertEqual(sum(r[2] for r in runs), len(shared))
        ranges = ranges_for_pair(self.st, ma, mb)
        self.assertEqual(len(ranges), 1)
        clipped = ranges[0][2]
        # the aligned interior of a 12 KiB run trims at most 4 KiB per side
        self.assertGreater(clipped, len(shared) - 2 * BLOCK)
        self.assertLessEqual(clipped, len(shared))
        for off_a, off_b, ln in ranges:
            self.assertEqual(off_a % BLOCK, 0)
            self.assertEqual(off_b % BLOCK, 0)
            self.assertEqual(ln % BLOCK, 0)

    def test_phase_mismatched_shared_run_is_skipped_not_misaligned(self):
        # Same construction but the start delta is 51: FIDEDUPERANGE cannot
        # map identical content across those phases, so the range list must
        # be empty rather than carrying misaligned offsets.
        shared = os.urandom(3 * BLOCK)
        pa = self._mk("ra.gguf", [(len(shared), shared)])
        pb = self._mk("rb.gguf", [(51, os.urandom(51)),
                                  (len(shared), shared)])
        ma = self.st.load_manifest(self.st.add(pa))
        mb = self.st.load_manifest(self.st.add(pb))
        self.assertNotEqual((ma["tensors"][0]["start"] - mb["tensors"][1]["start"]) % BLOCK, 0)
        runs = find_shared_runs(manifest_file_chunks(self.st, ma),
                                manifest_file_chunks(self.st, mb))
        self.assertEqual(sum(r[2] for r in runs), len(shared))
        self.assertEqual(ranges_for_pair(self.st, ma, mb), [])

    def test_bind_file_matches_manifest_by_header_hash(self):
        p = self._mk("b.gguf", [(1024, os.urandom(1024))])
        q = self._mk("c.gguf", [(1024, os.urandom(1024))])
        mid = self.st.add(p)
        qid = self.st.load_manifest(self.st.add(q))["id"]
        self.assertEqual(bind_file(self.st, p)["id"], mid)
        self.assertEqual(bind_file(self.st, q)["id"], qid)


class IoctlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch_dir()
        self.addCleanup(_rmtree, self.tmp)
        ok, note = fs_supports_dedupe(self.tmp)
        if not ok:
            self.skipTest(f"FIDEDUPERANGE unsupported here: {note}")

    def test_identical_files_share_extents_and_keep_bytes(self):
        c = 1 << 20
        pa = os.path.join(self.tmp, "x.gguf")
        payload = os.urandom(2 * c)
        with open(pa, "wb") as f:
            f.write(build_gguf(tensors=[("t0", [2 * c // 64, 64], 0, payload)]))
        pb = os.path.join(self.tmp, "y.gguf")
        with open(pa, "rb") as f, open(pb, "wb") as g:
            g.write(f.read())
        sha_a, sha_b = sha256_file(pa), sha256_file(pb)
        self.assertEqual(sha_a, sha_b)

        st = Store(os.path.join(self.tmp, "store"), chunk_size=c)
        self.addCleanup(st.close)
        man = st.load_manifest(st.add(pb))
        self.assertEqual(bind_file(st, pa)["id"], man["id"])
        self.assertEqual(bind_file(st, pb)["id"], man["id"])
        ranges = ranges_for_pair(st, man, man)
        planned = sum(r[2] for r in ranges)
        # the aligned interior of the payload region
        start = man["tensors"][0]["start"]
        self.assertEqual(planned, (start + 2 * c) // BLOCK * BLOCK - (start + BLOCK - 1) // BLOCK * BLOCK)

        fd_a = os.open(pa, os.O_RDONLY)
        fd_b = os.open(pb, os.O_RDONLY)
        try:
            applied = 0
            for off_a, off_b, ln in ranges:
                mapped, note = fideduperange(fd_a, off_a, fd_b, off_b, ln)
                self.assertIsNone(note)
                applied += mapped
            self.assertEqual(applied, planned)
            # XFS st_blocks counts logical extents per inode, so sharing
            # shows no shrink.  Prove the share via FIEMAP instead: the
            # payload extents must back both files identically and be
            # flagged shared.
            for off_a, off_b, ln in ranges:
                ea = physical_extents(fd_a, off_a, ln)
                eb = physical_extents(fd_b, off_b, ln)
                self.assertEqual([(lo, ph, n) for lo, ph, n, _ in ea],
                                 [(lo, ph, n) for lo, ph, n, _ in eb])
                self.assertTrue(ea)
                for _, _, _, flags in eb:
                    self.assertTrue(flags & FIEMAP_EXTENT_SHARED)
        finally:
            os.close(fd_a)
            os.close(fd_b)
        self.assertEqual(sha256_file(pa), sha_a)
        self.assertEqual(sha256_file(pb), sha_a)

    def test_identical_interior_at_shifted_offsets(self):
        shared = os.urandom(512 * 1024)
        pa = os.path.join(self.tmp, "pa.gguf")
        pb = os.path.join(self.tmp, "pb.gguf")
        with open(pa, "wb") as f:
            f.write(build_gguf(tensors=[("t0", [c // 64 for c in (len(shared),)], 0, shared)]))
        tail = os.urandom(64 * 1024)
        with open(pb, "wb") as f:
            f.write(build_gguf(tensors=[("t0", [len(shared) // 64, 64], 0, shared + tail)]))
        sha_b = sha256_file(pb)
        fd_a = os.open(pa, os.O_RDONLY)
        fd_b = os.open(pb, os.O_RDONLY)
        try:
            applied, note = fideduperange(fd_a, 0, fd_b, 0, len(shared))
            self.assertIsNone(note)
            self.assertEqual(applied, len(shared) // BLOCK * BLOCK)
        finally:
            os.close(fd_a)
            os.close(fd_b)
        self.assertEqual(sha256_file(pb), sha_b)


if __name__ == "__main__":
    unittest.main()