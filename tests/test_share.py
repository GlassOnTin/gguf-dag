"""Weight-pool tooling: registry protocol, size table, manifest generation."""

import hashlib
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_dag.ggml import payload_nbytes
from gguf_dag.share import (MAGIC, VERSION, QUERY, ST_HIT, ST_MISS,
                            ShareError, RegistryClient, normalize_bus,
                            parse_share_manifest, render_share_manifest,
                            expand_shards, ensure_manifest, file_state,
                            _digests_from_file, _digests_from_store)
from gguf_dag.store import Store
import util as gguf_util


HDR = struct.Struct("<6IQ32s")


class FakeRegistry:
    """Tiny in-process registry double: HIT for preloaded digests, else MISS.

    Runs one accepted connection per query, like the real daemon's protocol."""

    def __init__(self, path, segments=None):
        self.path = str(path)
        self.segments = segments or {}   # digest32 -> (seg_size, bus)
        self.requests = []               # (digest, bus) of queries seen
        self.sock = None
        self.thread = None

    def __enter__(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.sock.bind(self.path)
        self.sock.listen(8)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *a):
        if self.sock:
            self.sock.close()
            self.thread.join(timeout=2)
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    raw = self._recv(conn, HDR.size)
                    magic, ver, _t, _s, _nl, bus_len, _sz, digest = HDR.unpack(raw)
                    bus = self._recv(conn, bus_len)[:-1].decode() if bus_len else None
                    self.requests.append((digest, bus))
                    if magic == MAGIC and ver == VERSION and digest in self.segments:
                        seg_size, reg_bus = self.segments[digest]
                        reply = HDR.pack(MAGIC, VERSION, QUERY, ST_HIT, 0,
                                         len(reg_bus) + 1, seg_size, digest)
                        conn.sendall(reply + reg_bus.encode() + b"\0")
                    else:
                        reply = HDR.pack(MAGIC, VERSION, QUERY, ST_MISS, 0, 0, 0, digest)
                        conn.sendall(reply)
                except OSError:
                    return

    @staticmethod
    def _recv(conn, n):
        buf = bytearray()
        while len(buf) < n:
            piece = conn.recv(n - len(buf))
            if not piece:
                raise ConnectionError("client closed mid-request")
            buf.extend(piece)
        return bytes(buf)


class TestRegistryClient(unittest.TestCase):
    def test_hit_then_miss_two_connections(self):
        digest = b"\x01" * 32
        with tempfile.TemporaryDirectory() as tmp:
            with FakeRegistry(Path(tmp) / "r.sock", segments={digest: (4096, "0000:19:00.0")}) as reg:
                c = RegistryClient(Path(tmp) / "r.sock")
                self.assertEqual(c.query(digest, "0000:19:00.0")[0], "hit")
                self.assertEqual(c.query(b"\x02" * 32, "0000:19:00.0")[0], "miss")
                self.assertEqual(len(reg.requests), 2)
                seen_digest, seen_bus = reg.requests[0]
                self.assertEqual(seen_digest, digest)
                self.assertEqual(seen_bus, "0000:19:00.0")

    def test_digest_length_enforced(self):
        c = RegistryClient("/nonexistent.sock")
        with self.assertRaises(ShareError):
            c.query(b"short", "0000:49:00.0")

    def test_missing_registry_raises(self):
        c = RegistryClient("/nonexistent/gguf-dag-test.sock")
        with self.assertRaises(ShareError):
            c.query(b"\0" * 32, "0000:49:00.0")


class TestSizeTable(unittest.TestCase):
    def test_q8_0_ground_truth(self):
        # dims and published nbytes from a real qwen2.5-1.5B q8_0 tensor
        self.assertEqual(payload_nbytes([1536, 8960], 8), 14622720)

    def test_iq4_nl_and_iq3_xxs_ground_truth(self):
        # per_layer_token_embd.weight in Qwen3.8-Flash-Next-UD-Q4_K_XL (iq4_nl, type 20)
        self.assertEqual(payload_nbytes([3840, 248320], 20), (3840 // 32) * 18 * 248320)
        # iq3_xxs (type 18: 256-elem block, 98 bytes)
        self.assertEqual(payload_nbytes([512, 64], 18), (512 // 256) * 98 * 64)

    def test_scalars(self):
        self.assertEqual(payload_nbytes([256, 1024], 0), 1048576)   # f32
        self.assertEqual(payload_nbytes([512], 1), 1024)             # f16

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            payload_nbytes([256], 999)

    def test_unaligned_ne0_raises(self):
        with self.assertRaises(ValueError):
            payload_nbytes([1000], 8)  # q8_0 needs ne0 % 32 == 0


class TestManifestFormat(unittest.TestCase):
    def test_round_trip(self):
        entries = [("a.weight", "ab" * 32, 1088), ("b.weight", "cd" * 32, 4000)]
        text = render_share_manifest(1894532128, 1791826864, entries)
        self.assertTrue(text.startswith("#gguf-share-manifest\t1894532128\t1791826864\n"))
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write(text)
            f.flush()
            total, mtime, parsed = parse_share_manifest(f.name)
        os.unlink(f.name)
        self.assertEqual((total, mtime), (1894532128, 1791826864))
        self.assertEqual(parsed, entries)

    def test_malformed_entry_rejected(self):
        bad = "#gguf-share-manifest\t10\t5\na.weight\t" + "zz" * 32 + "\t5\n"
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write(bad)
            f.flush()
            with self.assertRaises(ShareError):
                parse_share_manifest(f.name)
        os.unlink(f.name)

    def test_normalize_bus(self):
        self.assertEqual(normalize_bus("00000000:49:00.0"), "0000:49:00.0")
        self.assertEqual(normalize_bus("0000:49:00.0"), "0000:49:00.0")
        self.assertEqual(normalize_bus("00000001:19:00.0"), "0001:19:00.0")


class TestDigests(unittest.TestCase):
    """Payload digests from real bytes, with padding in the spans."""

    def setUp(self):
        self.t1 = os.urandom(1024 // 32 * 34)    # q8_0 1024 elements
        self.t2 = os.urandom(1000 * 4)           # f32 1000 elements
        self.blob = gguf_util.build_gguf(
            tensors=[("t1", [1024], 8, self.t1), ("t2", [1000], 0, self.t2)],
            inter_gap=13, tail_pad=7)

    def test_from_file_hashes_payload_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.gguf"
            p.write_bytes(self.blob)
            entries = _digests_from_file(p)
        self.assertEqual([e[0] for e in entries], ["t1", "t2"])
        self.assertEqual([e[2] for e in entries], [len(self.t1), len(self.t2)])
        self.assertEqual(entries[0][1], hashlib.sha256(self.t1).hexdigest())
        self.assertEqual(entries[1][1], hashlib.sha256(self.t2).hexdigest())

    def test_from_store_matches_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.gguf"
            p.write_bytes(self.blob)
            with Store(Path(tmp) / "store") as st:
                mid = st.add(p)
                man = st.load_manifest(mid)
                entries = _digests_from_store(st, man)
        self.assertEqual([e[1] for e in entries], [hashlib.sha256(self.t1).hexdigest(),
                                                   hashlib.sha256(self.t2).hexdigest()])
        self.assertEqual(entries[0][2], len(self.t1))

    def test_multi_shard_manifest(self):
        # Shard 1: metadata only (0 tensors); Shard 2: t1 + t2
        s1_blob = gguf_util.build_gguf(tensors=[])
        with tempfile.TemporaryDirectory() as tmp:
            s1 = Path(tmp) / "model-00001-of-00002.gguf"
            s2 = Path(tmp) / "model-00002-of-00002.gguf"
            s1.write_bytes(s1_blob)
            s2.write_bytes(self.blob)
            self.assertEqual(expand_shards(s1), [str(s1), str(s2)])
            self.assertEqual(expand_shards(s2), [str(s1), str(s2)])
            mpath, entries, _src = ensure_manifest(s1)
            self.assertEqual(mpath, Path(str(s1) + ".share.tsv"))
            self.assertEqual([e[0] for e in entries], ["t1", "t2"])
            total, mtime, parsed = parse_share_manifest(mpath)
            self.assertEqual((total, mtime), file_state([s1, s2]))
            self.assertEqual(parsed, entries)

    def test_table_disagreement_raises(self):
        # dims claim 2048 f32 elements (8192 bytes) but the file holds less
        bad = gguf_util.build_gguf(tensors=[("x", [2048], 0, b"\x00" * 100),
                                            ("y", [8], 0, b"\x00" * 32)])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.gguf"
            p.write_bytes(bad)
            with self.assertRaises(ShareError):
                _digests_from_file(p)


if __name__ == "__main__":
    unittest.main()