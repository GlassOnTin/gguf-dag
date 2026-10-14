"""GGML tensor payload sizes.

The store records tensors as byte *spans* (payload plus any inter-tensor
padding), which is all byte-identical rebuild needs.  Weight sharing needs the
exact payload size of each tensor, which in GGUF is implied by the tensor's
ggml type and shape.  The table here mirrors the fork's ggml: the enumeration
from ggml/include/ggml.h, the block layout sizes from ggml/src/ggml-common.h
(static_assert sizeof(block_*)), and the blck sizes from the type_traits table
in ggml/src/ggml.c (branch @ deps/llama.cpp, ipc-share-vmm edd070144).

Types whose constants could not be read off those definitions are deliberately
absent: payload_nbytes() raises on ggml_type values outside the table rather
than guessing.  ggml_nbytes() semantics: payload = type_size * (ne[0] /
blck_size) * prod(ne[1:]), with ne[0] required to be divisible by blck_size
for quantized types (real GGUF converters always satisfy this; refusing to
guess keeps a wrong split from passing silently).
"""

# ggml_type enum value -> (blck_size, type_size)
_TYPE = {
    0:  (1, 4),     # F32
    1:  (1, 2),     # F16
    2:  (32, 18),   # Q4_0
    3:  (32, 20),   # Q4_1
    6:  (32, 22),   # Q5_0
    7:  (32, 24),   # Q5_1
    8:  (32, 34),   # Q8_0
    9:  (32, 36),   # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),   # MXFP4
    40: (64, 36),   # NVFP4 (one 64-element block: 4 UE4M3 scales + 32 packed nibbles)
}

_NAMES = {
    id: name for id, name in zip(_TYPE, (
        "f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1",
        "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K", "iq2_xxs", "iq2_xs",
        "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs", "i8", "i16",
        "i32", "i64", "f64", "iq1_m", "bf16", "tq1_0", "tq2_0", "mxfp4", "nvfp4"))
}


def type_name(ggml_type):
    """Human-readable type name (for messages only)."""
    return _NAMES.get(ggml_type, f"ggml_type {ggml_type}")


def payload_nbytes(dims, ggml_type):
    """Exact payload byte count of a tensor, as ggml_nbytes() would compute.

    dims is the GGUF ne[] list (ne[0] first).  Raises on unknown types and on
    quantized ne[0] that is not a multiple of the block size.
    """
    e = _TYPE.get(int(ggml_type))
    if e is None:
        raise ValueError(f"ggml type {ggml_type} is not in the size table; "
                         "add its block layout verified against ggml-common.h")
    blck, tsz = e
    if len(dims) < 1 or any(d < 1 for d in dims):
        raise ValueError(f"bad dims {dims}")
    if blck > 1 and dims[0] % blck:
        raise ValueError(f"ne[0]={dims[0]} is not a multiple of blck_size {blck} "
                         f"for {type_name(ggml_type)}; refusing to guess the payload size")
    n = (dims[0] // blck if blck > 1 else dims[0]) * tsz
    for d in dims[1:]:
        n *= d
    return n