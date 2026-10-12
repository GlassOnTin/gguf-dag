"""Integration tests on real GGUF files from the local model directory.

Skipped when files are absent (CI-safe). The parser cross-check asserts the
adjacent-offset span computation agrees with gguf-py's per-type n_bytes for
every tensor, which is what licenses the table-free design.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gguf_dag.gguf import GGUFFile, GGUFError
from gguf_dag.store import Store

MODELS = Path("/home/ian/models")
MMPROJ = MODELS / "Qwen3.8-Flash-Next/mmproj-BF16.gguf"
SHARD1 = MODELS / "Qwen3.8-Flash-Next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
GGUF_PY = MODELS / "llama.cpp/gguf-py"


def _gguf_py_reader(path):
    sys.path.insert(0, str(GGUF_PY))
    from gguf import GGUFReader
    return GGUFReader(str(path))


@unittest.skipUnless(MMPROJ.exists(), "no local mmproj GGUF")
class TestRealFiles(unittest.TestCase):
    def test_roundtrip_mmproj_byte_identical(self):
        import hashlib
        with tempfile.TemporaryDirectory() as sdir, tempfile.TemporaryDirectory() as tmp:
            with Store(sdir, chunk_size=1 << 20) as st:
                mid = st.add(MMPROJ)
                out = Path(tmp) / "mmproj.rebuilt.gguf"
                st.rebuild(mid, out)
                a = hashlib.sha256(MMPROJ.read_bytes()).hexdigest()  # 0.85 GiB
                b = hashlib.sha256(out.read_bytes()).hexdigest()
            self.assertEqual(a, b)

    @unittest.skipUnless(SHARD1.exists(), "no local header-only shard")
    def test_header_only_shard_roundtrip(self):
        with tempfile.TemporaryDirectory() as sdir, tempfile.TemporaryDirectory() as tmp:
            with Store(sdir) as st:
                mid = st.add(SHARD1)
                man = st.load_manifest(mid)
                self.assertEqual(man["tensors"], [])
                out = Path(tmp) / "shard1.rebuilt"
                st.rebuild(mid, out)
            self.assertEqual(out.read_bytes(), SHARD1.read_bytes())

    def test_parser_matches_gguf_py(self):
        try:
            reader = _gguf_py_reader(MMPROJ)
        except Exception:
            self.skipTest("gguf-py unavailable")
        gf = GGUFFile(MMPROJ)
        self.assertEqual(len(gf.tensors), len(reader.tensors))
        for mine, theirs in zip(gf.tensors, reader.tensors):
            self.assertEqual(mine["name"], theirs.name)
            self.assertEqual(mine["ggml_type"], int(theirs.tensor_type))
            self.assertEqual(mine["dims"], list(map(int, theirs.shape)))
            self.assertEqual(mine["n_bytes"], int(theirs.n_bytes),
                             f"span != n_bytes for {mine['name']}")


@unittest.skipUnless(SHARD1.exists(), "no local files")
class TestParserAgainstGgufPyMore(unittest.TestCase):
    FILES = [SHARD1,
             MODELS / "Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"]

    def test_spans_equal_n_bytes(self):
        if not GGUF_PY.exists():
            self.skipTest("gguf-py unavailable")
        for path in self.FILES:
            if not path.exists():
                continue
            reader = _gguf_py_reader(path)
            gf = GGUFFile(path)
            for mine, theirs in zip(gf.tensors, reader.tensors):
                self.assertEqual(mine["n_bytes"], int(theirs.n_bytes),
                                 f"{path.name}:{mine['name']}")


if __name__ == "__main__":
    unittest.main()