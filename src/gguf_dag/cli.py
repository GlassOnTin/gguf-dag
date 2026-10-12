"""gguf-dag: content-addressed, hash-chained packing of GGUF model files."""

import argparse
import hashlib
import json
import os
import time

from .gguf import GGUFError
from . import __version__
from .store import Store


def _fmt_gib(n):
    return f"{n / 2**30:.2f} GiB"


def _human_stats(s):
    return (
        f"files:            {s['n_files']}\n"
        f"source total:     {_fmt_gib(s['total_source_bytes'])}\n"
        f"pack (unique):    {_fmt_gib(s['pack_bytes'])}\n"
        f"dedup saving:     {s['dedup_saving'] * 100:.2f}%\n"
        f"chunk records:    {s['chunk_records']}\n"
        f"index size:       {s['overhead_index_pct']:.4f}% of pack"
    )


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gguf-dag", description=__doc__)
    ap.add_argument("--store", default=os.environ.get("GGUF_DAG_STORE", "./gguf-dag.store"),
                    help="store directory (default ./gguf-dag.store or $GGUF_DAG_STORE)")
    ap.add_argument("--chunk", type=int, default=65536, help="chunk size in bytes for adds (default 65536)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="split GGUF file(s) into the store")
    p_add.add_argument("paths", nargs="+")
    p_add.add_argument("--hmac-key", help="file whose bytes sign each manifest (HMAC-BLAKE2b)")

    p_re = sub.add_parser("rebuild", help="rebuild a byte-identical GGUF from manifest + chunks")
    p_re.add_argument("ref", help="manifest id (prefix ok)")
    p_re.add_argument("-o", "--out")

    p_ver = sub.add_parser("verify", help="recompute a manifest's hash chain ('all' for every manifest)")
    p_ver.add_argument("ref")
    p_ver.add_argument("--hmac-key", help="file whose bytes verify manifest HMACs")

    p_ls = sub.add_parser("ls", help="tensor table of a manifest")
    p_ls.add_argument("ref")

    sub.add_parser("stats", help="store totals")

    p_sha = sub.add_parser("sha256", help="print sha256 of a file (byte-identity checks)")
    p_sha.add_argument("path")

    args = ap.parse_args(argv)

    if args.cmd == "sha256":
        print(_sha256(args.path))
        return 0

    if args.cmd == "stats":
        with Store(args.store, args.chunk) as st:
            print(_human_stats(st.stats()))
        return 0

    if args.cmd == "add":
        key = open(args.hmac_key, "rb").read() if args.hmac_key else None
        with Store(args.store, args.chunk) as st:
            for p in args.paths:
                t0 = time.time()
                man = st.load_manifest(st.add(p, hmac_key=key))
                label = man["source"]["name"]
                size = man["source"]["size"]
                nu = man["chunks"]["new_unique_bytes"]
                total = man["chunks"]["count"]
                hit = 1 - nu / size if size else 1.0
                print(f"{label}: {size / 2**30:.2f} GiB, {total} chunks, "
                      f"new {nu / 2**30:.2f} GiB, shared-chunk hit {hit * 100:.2f}% "
                      f"([{time.time() - t0:.1f}s] id {man['id'][:12]})")
        return 0

    if args.cmd == "rebuild":
        with Store(args.store, args.chunk) as st:
            man = st.load_manifest(args.ref)
            label = man["source"]["name"]
            out = args.out or f"{label}.rebuilt"
            t0 = time.time()
            sha = st.rebuild(args.ref, out)
            print(f"{out}: {man['source']['size'] / 2**30:.2f} GiB rebuilt in {time.time() - t0:.1f}s, "
                  f"sha256 {sha}")
        return 0

    if args.cmd == "verify":
        key = open(args.hmac_key, "rb").read() if args.hmac_key else None
        ok = True
        with Store(args.store, args.chunk) as st:
            refs = ([m["id"] for m in st.list_manifests()] if args.ref == "all" else [args.ref])
            for r in refs:
                man = st.load_manifest(r)
                n, probs = st.verify(r, hmac_key=key)
                status = "OK" if not probs else "FAIL"
                print(f"{status} {man['source']['name']}: {n} chunks checked")
                for p in probs:
                    ok = False
                    print(f"     {p}")
        return 0 if ok else 1

    if args.cmd == "ls":
        with Store(args.store, args.chunk) as st:
            man = st.load_manifest(args.ref)
            print(f"{man['source']['name']}  id {man['id']}  {man['source']['size'] / 2**30:.2f} GiB  "
                  f"chunk {man['chunk_size']} B")
            for t in man["tensors"]:
                dims = "x".join(str(d) for d in t["dims"])
                print(f"  {t['name']:60s} type={t['ggml_type']:<3d} {t['n_bytes'] / 2**20:12.1f} MiB  "
                      f"{t['n_chunks']:7d} chunks  {dims}")
        return 0

    ap.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()