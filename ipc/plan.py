"""Build a CUDA IPC demo plan from a gguf-dag store.

plan.txt is a line protocol the C++ agents consume without a JSON parser:

  PACK <path>                 absolute path of chunks.pack (the copy source)
  ARENA_BYTES <n>             arena size in bytes
  PUBLISH_CHUNK <digest16hex> <pack_off> <size> <sha256> <sum64> <xor8>
  PUBLISH_END
  CONSUME_CHUNK <digest16hex> <pack_off> <size> <sha256> <sum64> <xor8>
  CONSUME_END

The publisher copies from the pack, not the original GGUF — the demo points
are "loader reads the CAS store" and "consumer attaches shared pages instead
of re-allocating". Per-chunk fingerprints (sha256, byte-sum, byte-xor) let
the consumer check integrity both on the host (D2H + sha256) and on the
device (reduction kernel), for shared *and* unique chunks.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np


def load_index(path):
    """digest bytes -> (pack_offset, byte_len) from the flat 32 B record index."""
    idx = {}
    with open(path, "rb") as f:
        while rec := f.read(32):
            if len(rec) != 32:
                break
            d, off, ln = struct.unpack_from("<16sQI", rec)
            idx[d] = (off, ln)
    return idx


def chunk_list(st, mid, idx):
    """[(digest_hex, pack_off, size)] in file order, resolved against the pack."""
    mdir = st.manifest_path(mid).parent
    man = json.loads((mdir / "manifest.json").read_text())
    refs = (mdir / "refs.bin").read_bytes()
    out = []
    for t in man["tensors"]:
        stream = refs[t["refs_off"]: t["refs_off"] + 16 * t["n_chunks"]]
        for k in range(t["n_chunks"]):
            d = stream[k * 16: k * 16 + 16]
            off, ln = idx.get(d) or (None, None)
            if off is None:
                raise SystemExit(f"chunk {d.hex()} absent from pack index")
            out.append((d.hex(), off, ln))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--con", required=True)
    ap.add_argument("--out", default="plan.txt")
    ap.add_argument("--arena-mib", type=int, default=1024)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gguf_dag.store import Store

    st = Store(args.store)
    try:
        idx = load_index(st.root / "chunks.idx")
        pub = chunk_list(st, args.pub, idx)
        con = chunk_list(st, args.con, idx)
        pack_abs = (st.root / "chunks.pack").resolve()
        # unique digests across both roles get fingerprints in one pass
        uniq = {}
        for tag, lst in (("PUBLISH", pub), ("CONSUME", con)):
            for hx, off, ln in lst:
                uniq[hx] = (off, ln)
        pack = open(pack_abs, "rb")
        sha = {}
        for hx, (off, ln) in sorted(uniq.items()):
            pack.seek(off)
            data = pack.read(ln)
            h = hashlib.sha256(data).hexdigest()
            nv = np.frombuffer(data, dtype=np.uint8)
            sb = int(nv.sum(dtype=np.uint64))
            xv = int(np.bitwise_xor.reduce(nv))
            sha[hx] = (h, sb, xv)
        pack.close()

        with open(args.out, "w") as f:
            f.write(f"PACK {pack_abs}\n")
            f.write(f"ARENA_BYTES {args.arena_mib * 1024 * 1024}\n")
            for tag, lst in (("PUBLISH", pub), ("CONSUME", con)):
                for hx, off, ln in lst:
                    h, sb, xv = sha[hx]
                    f.write(f"{tag}_CHUNK {hx} {off} {ln} {h} {sb} {xv}\n")
                f.write(f"{tag}_END\n")
        shared = {hx for hx, *_ in con} & {hx for hx, *_ in pub}
        con_sizes = {hx: ln for hx, _, ln in con}
        sh_bytes = sum(con_sizes.get(hx, 0) for hx in shared)
        print(f"plan: {len(pub)} pub chunks, {len(con)} con chunks, "
              f"{len(shared)} shared, {sh_bytes / 2**20:.0f} MiB shared")
    finally:
        st.close()


if __name__ == "__main__":
    main()