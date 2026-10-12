"""Synthetic GGUF builder for tests.

Produces valid GGUF v2/v3 files with arbitrary tensor payloads. Tensor data
is placed with optional random junk gaps between tensors to exercise the
adjacent-offset span computation (junk padding lives inside the preceding
tensor's span; a round-trip must preserve it byte-for-byte).
"""

import os
import struct


def s(x):
    b = x.encode("utf-8") if isinstance(x, str) else x
    return struct.pack("<Q", len(b)) + b


def kv_u32(key, val):
    return s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def kv_f32(key, val):
    return s(key) + struct.pack("<I", 6) + struct.pack("<f", val)


def kv_u64(key, val):
    return s(key) + struct.pack("<I", 10) + struct.pack("<Q", val)


def kv_str(key, val):
    return s(key) + struct.pack("<I", 8) + s(val)


def kv_arr_str(key, vals):
    body = struct.pack("<I", 8) + struct.pack("<Q", len(vals)) + b"".join(s(v) for v in vals)
    return s(key) + struct.pack("<I", 9) + body


def kv_arr_u32(key, vals):
    body = struct.pack("<I", 4) + struct.pack("<Q", len(vals)) + b"".join(struct.pack("<I", v) for v in vals)
    return s(key) + struct.pack("<I", 9) + body


def build_gguf(version=3, kvs=(), tensors=(), alignment=32, inter_gap=0, tail_pad=0, rng=None):
    """tensors: list of (name, dims, ggml_type, payload_bytes).

    inter_gap > 0 inserts that many junk bytes before each tensor after the
    first; tail_pad appends junk after the last tensor (part of its span).
    """
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kvs))
    for k in kvs:
        header += k
    infos = b""
    # layout data section: first tensor at offset 0, then gaps
    offsets = []
    pos = 0
    for i, (name, dims, ttype, payload) in enumerate(tensors):
        if i > 0:
            pos += inter_gap
        offsets.append(pos)
        pos += len(payload)
    for (name, dims, ttype, payload), off in zip(tensors, offsets):
        infos += s(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) \
            + struct.pack("<IQ", ttype, off)
    if version >= 3:
        infos += b""  # (v3 adds kv types u64/i64/f64 only; layout identical here)
    data_start = (len(header) + len(infos) + alignment - 1) // alignment * alignment
    # metadata must end aligned; pad region between infos and data_start
    out = bytearray(header + infos)
    out += b"\x00" * (data_start - len(out))
    for i, ((name, dims, ttype, payload), off) in enumerate(zip(tensors, offsets)):
        if i > 0 and inter_gap:
            out += os.urandom(inter_gap)
        out += payload
    out += os.urandom(tail_pad) if tail_pad else b""
    return bytes(out)