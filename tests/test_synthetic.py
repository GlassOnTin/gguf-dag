"""Synthetic-file tests: parse spans, round-trip identity, dedup, corruption."""

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_dag.gguf import GGUFFile, GGUFError
from gguf_dag.store import Store
import util


def w(tmp, name, data):
    p = Path(tmp) / name
    p.write_bytes(data)
    return p


class TestParse(unittest.TestCase):
    def test_span_layout_with_gaps(self):
        t1 = os.urandom(1 << 20)
        t2 = os.urandom(100)
        t3 = os.urandom(70000)
        blob = util.build_gguf(
            kvs=[util.kv_str("general.architecture", "test"),
                 util.kv_u32("general.alignment", 32),
                 util.kv_arr_str("tokenizer.ggml.tokens", ["x", "yy", "zzz"])],
            tensors=[("t1", [256, 1024], 0, t1), ("t2", [25], 0, t2), ("t3", [17500], 0, t3)],
            inter_gap=13, tail_pad=7)
        with tempfile.TemporaryDirectory() as tmp:
            p = w(tmp, "a.gguf", blob)
            gf = GGUFFile(p)
            self.assertEqual([t["name"] for t in gf.tensors], ["t1", "t2", "t3"])
            # gap before t2 belongs to t1's span; gap before t3 to t2; tail_pad to t3
            self.assertEqual(gf.tensors[0]["n_bytes"], len(t1) + 13)
            self.assertEqual(gf.tensors[1]["n_bytes"], len(t2) + 13)
            self.assertEqual(gf.tensors[2]["n_bytes"], len(t3) + 7)
            self.assertEqual(gf.header_bytes, blob[: gf.data_start])

    def test_zero_tensor_file(self):
        blob = util.build_gguf(kvs=[util.kv_u64("some.count", 5)], tensors=[])
        with tempfile.TemporaryDirectory() as tmp:
            gf = GGUFFile(w(tmp, "h.gguf", blob))
            self.assertEqual(gf.tensors, [])
            self.assertEqual(gf.header_bytes, blob)

    def test_alignment_64(self):
        blob = util.build_gguf(kvs=[util.kv_u32("general.alignment", 64)],
                               tensors=[("a", [10], 0, os.urandom(33))])
        with tempfile.TemporaryDirectory() as tmp:
            gf = GGUFFile(w(tmp, "a.gguf", blob))
            self.assertEqual(gf.alignment, 64)
            self.assertEqual(gf.data_start % 64, 0)

    def test_rejects_non_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = w(tmp, "junk.gguf", b"\x00" * 6)
            with self.assertRaises(GGUFError):
                GGUFFile(p)

    def test_rejects_unordered_offsets(self):
        # rewrite the two tensor-info offsets so the first points past the second
        blob = bytearray(util.build_gguf(
            kvs=[util.kv_u32("general.alignment", 32)],
            tensors=[("a", [1], 7, b"x" * 4096), ("b", [1], 7, b"y" * 4096)]))
        i1 = blob.find(struct.pack("<IQ", 7, 0))        # info 'a' tail: ttype 7, offset 0
        i2 = blob.find(struct.pack("<IQ", 7, 4096))     # info 'b' tail: ttype 7, offset 4096
        self.assertGreater(i1, 0)
        self.assertGreater(i2, i1)
        blob[i1 + 4:i1 + 12] = struct.pack("<Q", 8192)  # first tensor starts past second
        blob[i2 + 4:i2 + 12] = struct.pack("<Q", 0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GGUFError):
                GGUFFile(w(tmp, "bad.gguf", bytes(blob)))


class TestRoundTrip(unittest.TestCase):
    def _roundtrip(self, blob, chunk=65536, hmac_key=None):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            p = w(tmp, "m.gguf", blob)
            with Store(sdir, chunk) as st:
                mid = st.add(p, hmac_key=hmac_key)
                out = Path(tmp) / "rebuilt.gguf"
                st.rebuild(mid, out)
            self.assertEqual(out.read_bytes(), blob)
            return mid

    def test_roundtrip_gaps_and_tail(self):
        blobs = os.urandom(1 << 20), os.urandom(100), os.urandom(70000)
        blob = util.build_gguf(
            kvs=[util.kv_str("general.architecture", "test"),
                 util.kv_arr_str("tokenizer.ggml.tokens", ["x", "yy", "zzz"]),
                 util.kv_arr_u32("scores", [1, 2, 3]),
                 util.kv_f32("f", 1.5),
                 util.kv_u64("u", 1 << 63)],
            tensors=[("t1", [256, 1024], 0, blobs[0]),
                     ("t2", [25], 0, blobs[1]),
                     ("t3", [17500], 0, blobs[2])],
            inter_gap=13, tail_pad=7)
        self._roundtrip(blob)

    def test_roundtrip_zero_tensors_and_v2(self):
        self._roundtrip(util.build_gguf(version=2, kvs=[util.kv_u64("c", 5)], tensors=[]))
        self._roundtrip(util.build_gguf(kvs=[], tensors=[]))

    def test_roundtrip_hmac(self):
        blob = util.build_gguf(kvs=[util.kv_str("a", "b")], tensors=[("t", [4], 0, os.urandom(4096))])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            p = w(tmp, "m.gguf", blob)
            with Store(sdir) as st:
                mid = st.add(p, hmac_key=b"secret")
                n, probs = st.verify(mid, hmac_key=b"secret")
                self.assertEqual(probs, [])
                out = Path(tmp) / "r.gguf"
                st.rebuild(mid, out)
            self.assertEqual(out.read_bytes(), blob)

    def test_hmac_tamper_detected(self):
        blob = util.build_gguf(tensors=[("t", [4], 0, os.urandom(4096))])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            p = w(tmp, "m.gguf", blob)
            with Store(sdir) as st:
                mid = st.add(p, hmac_key=b"secret")
                mj = st.manifest_path(mid)
                man = __import__("json").loads(mj.read_text())
                man["tensors"][0]["n_bytes"] += 8
                mj.write_text(__import__("json").dumps(man))
                n, probs = st.verify(mid, hmac_key=b"secret")
            self.assertTrue(any("HMAC" in p for p in probs))

    def test_dedup_across_files(self):
        shared = os.urandom(65536 + 100)
        A = util.build_gguf(kvs=[util.kv_str("general.architecture", "A")],
                            tensors=[("x", [1], 0, os.urandom(131072)), ("y", [1], 0, shared)])
        B = util.build_gguf(kvs=[util.kv_str("general.architecture", "B")],
                            tensors=[("y", [1], 0, shared)])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            with Store(sdir) as st:
                st.add(w(tmp, "a.gguf", A))
                man = st.load_manifest(st.add(w(tmp, "b.gguf", B)))
            size_b = B.__len__()
            new_b = man["chunks"]["new_unique_bytes"]
            # B's payload unique share = only its header + none of shared
            self.assertLess(new_b, size_b)
            with Store(sdir) as st2:
                stats = st2.stats()
            self.assertEqual(stats["new_unique_bytes_sum"], stats["pack_bytes"])

    def test_corruption_detected(self):
        blob = util.build_gguf(tensors=[("t", [256], 0, os.urandom(1 << 20))])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            with Store(sdir) as st:
                mid = st.add(w(tmp, "m.gguf", blob))
            with open(Path(sdir) / "chunks.pack", "r+b") as f:
                f.seek(17)
                b = f.read(1)
                f.seek(17)
                f.write(bytes([b[0] ^ 0xFF]))
            with Store(sdir) as st2:
                n, probs = st2.verify(mid)
            self.assertTrue(any("hash mismatch" in p for p in probs))

    def test_refs_corruption_detected(self):
        blob = util.build_gguf(tensors=[("t", [4], 0, os.urandom(4096))])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            with Store(sdir) as st:
                mid = st.add(w(tmp, "m.gguf", blob))
                mj = st.manifest_path(mid).parent
            refs = (mj / "refs.bin").read_bytes()
            (mj / "refs.bin").write_bytes(refs[:-1] + bytes([refs[-1] ^ 1]))
            with Store(sdir) as st2:
                n, probs = st2.verify(mid)
            self.assertTrue(any("refs" in p for p in probs))

    def test_chunk_size_option(self):
        blob = util.build_gguf(tensors=[("t", [4], 0, os.urandom(300000)), ("u", [4], 0, os.urandom(1000))])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as sdir:
            with Store(sdir, chunk_size=1 << 20) as st:
                mid = st.add(w(tmp, "m.gguf", blob))
                self.assertEqual(st.load_manifest(mid)["chunk_size"], 1 << 20)
                self.assertTrue(all(t["n_chunks"] == 1 for t in st.load_manifest(mid)["tensors"]))
                out = Path(tmp) / "r.gguf"
                st.rebuild(mid, out)
            self.assertEqual(out.read_bytes(), blob)


if __name__ == "__main__":
    unittest.main()