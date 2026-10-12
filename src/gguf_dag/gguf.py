"""Minimal structural walk of GGUF files.

Computes tensor *spans* (byte ranges containing the tensor payload plus any
inter-tensor padding) without a per-quant-type size table: span N ends where
tensor N+1 starts, and the final span extends to end of file. New GGML quant
types therefore need no parser changes.

The header region (everything before the first tensor byte) is returned as
one blob; rebuilding a byte-identical file is exactly header-blob + spans.
"""

import os
import struct

GGUF_MAGIC = b"GGUF"
VT_U8 = 0
VT_I8 = 1
VT_U16 = 2
VT_I16 = 3
VT_U32 = 4
VT_I32 = 5
VT_F32 = 6
VT_BOOL = 7
VT_STRING = 8
VT_ARRAY = 9
VT_U64 = 10
VT_I64 = 11
VT_F64 = 12
_SCALAR_FMT = {
    VT_U8: "B", VT_I8: "b", VT_U16: "H", VT_I16: "h",
    VT_U32: "I", VT_I32: "i", VT_F32: "f", VT_BOOL: "B",
    VT_U64: "Q", VT_I64: "q", VT_F64: "d",
}
_SCALAR_SZ = {k: struct.calcsize("<" + f) for k, f in _SCALAR_FMT.items()}

_MAX_KLEN = 1 << 20   # metadata keys and tensor names; real ones are far shorter
_MAX_NEED = 1 << 26   # no single refill may fetch more than 64 MiB
_RETRIES = 1 << 16    # per-construct refill attempts before declaring corruption


class GGUFError(ValueError):
    pass


def align_up(x, a):
    return (x + a - 1) // a * a


class GGUFFile:
    """Parsed structure of one GGUF file."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.file_size = os.path.getsize(self.path)
        self.version = None
        self.alignment = 32
        self.data_start = None
        self.tensors = []  # dicts: name, ggml_type, dims, start, n_bytes, offset
        self.header_bytes = None
        self._parse()

    def _parse(self):
        buf = bytearray()
        off = 24

        def need(n):
            """Extend buf by at least n bytes (always reading 64 KiB chunks).

            Returns False only on end-of-file.  n beyond _MAX_NEED means a
            length field was misread and the cursor has desynced from the
            file: refuse rather than allocate a file-sized buffer (this used
            to read ~file_size bytes in one call from a jumped offset).
            """
            n = int(n)
            if n > _MAX_NEED:
                raise GGUFError(
                    f"{self.path}: header parse wants {n} more bytes at offset {off}; "
                    "the file is corrupt or the parser desynced")
            got = self._fh.read(max(n, 1 << 16))
            buf.extend(got)
            return bool(got)

        self._fh = open(self.path, "rb")
        try:
            need(24)
            if len(buf) < 24 or bytes(buf[:4]) != GGUF_MAGIC:
                raise GGUFError(f"{self.path}: not a GGUF file")
            self.version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
            if self.version not in (2, 3):
                raise GGUFError(f"{self.path}: unsupported GGUF version {self.version}")
            while n_kv:
                while len(buf) - off < 12:
                    if not need(12 - (len(buf) - off)):
                        raise GGUFError(f"{self.path}: unexpected EOF in metadata keys at offset {off}")
                (klen,) = struct.unpack_from("<Q", buf, off)
                if klen > _MAX_KLEN:
                    raise GGUFError(
                        f"{self.path}: metadata key length {klen} at offset {off} "
                        f"exceeds {_MAX_KLEN}")
                off += 8
                while len(buf) - off < klen + 4:
                    if not need(klen + 4 - (len(buf) - off)):
                        raise GGUFError(f"{self.path}: unexpected EOF in metadata at offset {off}")
                key = bytes(buf[off:off + klen]).decode("utf-8", "replace")
                off += klen
                (vt,) = struct.unpack_from("<I", buf, off)
                off += 4
                # _value_end is side-effect-free, so retries restart safely.
                for _ in range(_RETRIES):
                    try:
                        end = self._value_end(buf, off, vt, self.file_size)
                        break
                    except (struct.error, IndexError):
                        if not need(1 << 16):
                            raise GGUFError(
                                f"{self.path}: unexpected EOF in value of '{key}' at offset {off}")
                else:
                    raise GGUFError(f"{self.path}: value of '{key}' at offset {off} did not resolve")
                if key == "general.alignment" and vt == VT_U32:
                    (self.alignment,) = struct.unpack_from("<I", buf, off)
                off = end
                n_kv -= 1
            infos = []
            # Parse each tensor info through a local cursor and commit only on
            # success: any mid-body failure (e.g. a length field spanning the
            # buffered boundary) must retry from the loop's own start offset,
            # never from whatever `off` got advanced to.
            for _ in range(n_tensors):
                for _ in range(_RETRIES):
                    try:
                        o = off
                        name, o = self._read_str(buf, o)
                        (nd,) = struct.unpack_from("<I", buf, o)
                        if nd > 32:
                            raise GGUFError(
                                f"{self.path}: tensor '{name}' has nd {nd} at offset {o}; "
                                f"GGML tensors have at most a few dimensions")
                        o += 4
                        if len(buf) - o < 8 * nd + 12:
                            if not need(8 * nd + 12 - (len(buf) - o)):
                                raise GGUFError(f"{self.path}: unexpected EOF at offset {o}")
                        dims = struct.unpack_from("<" + "Q" * nd, buf, o)
                        o += 8 * nd
                        ttype, toff = struct.unpack_from("<IQ", buf, o)
                        o += 12
                        off = o
                        break
                    except (struct.error, IndexError):
                        if not need(1 << 16):
                            raise GGUFError(f"{self.path}: unexpected EOF at offset {off}")
                else:
                    raise GGUFError(
                        f"{self.path}: tensor info at offset {off} did not resolve after {_RETRIES} refills")
                infos.append((name, list(dims), ttype, toff))
            self.data_start = align_up(off, self.alignment)
            while len(buf) < self.data_start:
                if not need(self.data_start - len(buf)):
                    raise GGUFError(f"{self.path}: unexpected EOF before the data section")
            if self.data_start > self.file_size:
                raise GGUFError(
                    f"{self.path}: header/pad ends at {self.data_start} beyond file size {self.file_size}")
            self.header_bytes = bytes(buf[: self.data_start])
            prev = None
            for i, (name, dims, ttype, rel_off) in enumerate(infos):
                start = self.data_start + rel_off
                if start >= self.file_size and not (i == len(infos) - 1 and start == self.file_size):
                    raise GGUFError(f"{self.path}: tensor '{name}' offset {rel_off} past data section")
                if prev is not None and start < prev:
                    raise GGUFError(
                        f"{self.path}: tensor offsets not strictly increasing; "
                        "adjacent-offset span computation requires ordered tensor infos")
                prev = start
                self.tensors.append({
                    "name": name, "ggml_type": ttype, "dims": list(dims),
                    "start": start, "offset": rel_off,
                })
            for i, t in enumerate(self.tensors):
                t["n_bytes"] = (self.tensors[i + 1]["start"] if i + 1 < len(self.tensors)
                                else self.file_size) - t["start"]
        finally:
            self._fh.close()

    def _read_str(self, buf, off):
        (n,) = struct.unpack_from("<Q", buf, off)
        if n > _MAX_KLEN:
            raise GGUFError(f"string length {n} at offset {off} exceeds {_MAX_KLEN}")
        if len(buf) - off - 8 < n:
            raise struct.error("string not yet buffered")
        return bytes(buf[off + 8: off + 8 + n]).decode("utf-8", "replace"), off + 8 + n

    @staticmethod
    def _value_end(buf, off, vt, fsize):
        """End offset of a metadata value; every extent checked against fsize."""
        if vt in _SCALAR_SZ:
            return off + _SCALAR_SZ[vt]
        if vt == VT_STRING:
            (n,) = struct.unpack_from("<Q", buf, off)
            if off + 8 + n > fsize:
                raise GGUFError(f"string value of {n} bytes at offset {off} exceeds file size")
            return off + 8 + n
        if vt == VT_ARRAY:
            et, cnt = struct.unpack_from("<IQ", buf, off)
            off += 12
            if et in _SCALAR_SZ:
                if off + _SCALAR_SZ[et] * cnt > fsize:
                    raise GGUFError(f"{cnt}-element array at offset {off} exceeds file size")
                return off + _SCALAR_SZ[et] * cnt
            if et == VT_STRING:
                for _ in range(cnt):
                    (n,) = struct.unpack_from("<Q", buf, off)
                    if off + 8 + n > fsize:
                        raise GGUFError(f"string of {n} bytes at offset {off} exceeds file size")
                    if len(buf) - off - 8 < n:
                        raise struct.error("string array not yet buffered")
                    off += 8 + n
                return off
            raise GGUFError(f"unsupported nested array element type {et}")
        raise GGUFError(f"unsupported metadata value type {vt}")