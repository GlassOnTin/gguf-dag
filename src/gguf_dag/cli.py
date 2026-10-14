"""gguf-dag: content-addressed, hash-chained packing of GGUF model files."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .gguf import GGUFError
from . import __version__
from .dedupe import bind_file, fideduperange, ranges_for_pair
from .share import (ShareError, RegistryClient, ensure_manifest, probe_buses,
                    normalize_bus, store_tensor_digests, fresh_publish_manifest,
                    parse_share_manifest, file_state, default_manifest,
                    expand_shards)
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


def _share_manifest(args, gguf):
    """Manifest entries for `share status`: read-only resolution.

    Publish-run manifest next to the file first, then an existing generated
    one (flagging a stale gate), then the store digest cache — computing the
    cache is allowed but costs a full hash pass, which status notes.
    """
    shards = expand_shards(gguf)
    fresh = fresh_publish_manifest(gguf)
    if fresh is not None:
        _t, _m, entries = parse_share_manifest(fresh)
        return entries, f"publish-run manifest {os.path.basename(fresh)}"
    gen = default_manifest(gguf)
    if os.path.exists(gen):
        total, mtime, entries = parse_share_manifest(gen)
        note = ""
        if (total, mtime) != file_state(shards):
            note = " (stale gate: the file changed after generation)"
        return entries, f"generated manifest {os.path.basename(gen)}{note}"
    if os.path.exists(args.store):
        with Store(args.store, args.chunk) as st:
            mans = [bind_file(st, s) for s in shards]
            if all(m is not None for m in mans):
                entries = []
                for m in mans:
                    entries.extend(store_tensor_digests(st, m))
                ids = ",".join(m["id"][:12] for m in mans)
                print(f"(no manifest yet; computing store digests for {ids}, "
                      "cached afterwards)")
                return entries, f"store manifest{'s' if len(mans) > 1 else ''} {ids}"
    raise ShareError(
        f"no share manifest for {gguf}: none next to the file, and the file is "
        "not in the store. Run 'gguf-dag share gen' or add it to the store first.")


def _share_prewarm_bin(args):
    bin_ = getattr(args, "share_prewarm_bin", None)
    if not bin_:
        pattern = "deps/llama.cpp/build*/bin/llama-share-prewarm"
        bases = [Path.cwd(), Path(__file__).resolve().parents[2]]
        for base in bases:
            hits = sorted(base.glob(pattern))
            if hits:
                bin_ = str(hits[-1])
                break
    if not bin_ or not os.path.exists(bin_):
        raise ShareError(
            "llama-share-prewarm not found; build the fork submodule "
            "(cmake -DGGML_CUDA=ON, target llama-share-prewarm) or pass "
            "--share-prewarm-bin")
    return bin_


def _share_store_root(args):
    """Store root or None (file-direct fallback), with a note on the downgrade."""
    if os.path.isdir(args.store):
        return args.store
    print(f"(store {args.store} does not exist; hashing from the GGUF file directly)")
    return None


def _share_cmd_gen(args):
    path, entries, source = ensure_manifest(args.ggf, _share_store_root(args),
                                            out=args.out, force=args.force)
    total = sum(b for _n, _d, b in entries)
    print(f"manifest {path}: {len(entries)} tensors, {total / 2**20:.1f} MiB payload — source: {source}")


def _share_cmd_status(args):
    socket_path = args.share_socket or os.environ.get("GGUF_DAG_SHARE_SOCKET")
    if not socket_path:
        raise ShareError("no registry socket: pass --share-socket or set $GGUF_DAG_SHARE_SOCKET")
    entries, source = _share_manifest(args, args.ggf)
    buses = [normalize_bus(b) for b in args.share_bus] if args.share_bus else probe_buses()
    if not buses:
        raise ShareError("no PCI bus id: pass --share-bus (or make nvidia-smi available)")
    client = RegistryClient(socket_path)
    digests = [bytes.fromhex(d) for _n, d, _b in entries]
    seg_total = 0
    per_bus_hits = {b: 0 for b in buses}
    hit = [False] * len(digests)
    for i, d in enumerate(digests):
        for b in buses:
            s, seg, _got = client.query(d, b)
            if s == "hit":
                hit[i] = True
                seg_total += seg
                per_bus_hits[b] += 1
    n = sum(hit)
    print(f"{os.path.basename(args.ggf)}: {n}/{len(entries)} tensors resident "
          f"({seg_total / 2**20:.1f} MiB of segments), manifest: {source}")
    for b in buses:
        print(f"  bus {b}: {per_bus_hits[b]}/{len(entries)}")
    return 0 if n == len(entries) else 1


def _share_cmd_warm(args):
    socket_path = args.share_socket or os.environ.get("GGUF_DAG_SHARE_SOCKET")
    if not socket_path:
        raise ShareError("no registry socket: pass --share-socket or set $GGUF_DAG_SHARE_SOCKET")
    if not os.path.exists(socket_path):
        raise ShareError(f"registry socket {socket_path} does not exist; start llama-share-registry first")
    prewarm = _share_prewarm_bin(args)
    mpath = args.share_manifest
    if mpath is None:
        path, entries, source = ensure_manifest(args.ggf, _share_store_root(args))
        mpath = str(path)
        total = sum(b for _n, _d, b in entries)
        print(f"manifest {mpath}: {len(entries)} tensors, {total / 2**20:.1f} MiB — source: {source}")
    shards = expand_shards(args.ggf)
    t0 = time.time()
    rc = 0
    for s in shards:
        r = subprocess.run([prewarm, socket_path, mpath, s])
        if r.returncode != 0:
            rc = r.returncode
            break
    print(f"prewarm exit {rc} in {time.time() - t0:.2f}s "
          f"({os.path.basename(args.ggf)}, {len(shards)} shard(s), socket {socket_path})")
    return rc


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

    p_ded = sub.add_parser("dedupe", help="FIDEDUPERANGE files in place, guided by their manifests")
    p_ded.add_argument("paths", nargs="+", help=".gguf files or directories to recurse")
    p_ded.add_argument("--dry-run", action="store_true", help="report planned sharing without applying")
    p_ded.add_argument("--min-gib", type=float, default=0.1,
                       help="skip pairs whose expected shared bytes are below this (GiB)")

    p_share = sub.add_parser("share", help="cross-process weight pool: manifests, registry, prewarm")
    share_sub = p_share.add_subparsers(dest="share_cmd", required=True)
    p_sg = share_sub.add_parser("gen", help="generate a share manifest (<gguf>.share.tsv by default)")
    p_sg.add_argument("ggf", help="the .gguf file")
    p_sg.add_argument("--out", help="write to this path instead of <gguf>.share.tsv")
    p_sg.add_argument("--force", action="store_true",
                      help="overwrite an existing manifest whose gate no longer matches")
    p_ss = share_sub.add_parser("status", help="query the registry: how much of a model is resident")
    p_ss.add_argument("ggf")
    p_ss.add_argument("--share-socket", help="registry socket (default $GGUF_DAG_SHARE_SOCKET)")
    p_ss.add_argument("--share-bus", action="append",
                      help="PCI bus id to query under (default: nvidia-smi probe)")
    p_sw = share_sub.add_parser("warm", help="seed the registry pool for a model via llama-share-prewarm")
    p_sw.add_argument("ggf")
    p_sw.add_argument("--share-socket", help="registry socket (default $GGUF_DAG_SHARE_SOCKET)")
    p_sw.add_argument("--share-manifest", help="use an existing manifest without generating")
    p_sw.add_argument("--share-prewarm-bin", dest="share_prewarm_bin",
                      help="llama-share-prewarm binary (default: deps/llama.cpp/build*/bin/llama-share-prewarm)")

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
                # the header region is stored verbatim, never chunked
                payload = size - man["header"]["raw_bytes"]
                hit = 1 - nu / payload if payload > 0 else 1.0
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

    if args.cmd == "dedupe":
        files = []
        for p in args.paths:
            if os.path.isdir(p):
                for base, _, names in os.walk(p):
                    files.extend(os.path.join(base, n) for n in sorted(names) if n.endswith(".gguf"))
            else:
                files.append(p)
        files = sorted(set(files))
        bound = []
        with Store(args.store, args.chunk) as st:
            for f in files:
                man = bind_file(st, f)
                if man is None:
                    print(f"unbindable (no matching manifest): {f}")
                else:
                    bound.append((f, man))
            if len(bound) < 2:
                print(f"{len(bound)} bound files, need at least two")
                return 1
            pairs = []
            for (fa, ma) in bound:
                for (fb, mb) in bound:
                    if fa >= fb:
                        continue
                    ranges = ranges_for_pair(st, ma, mb)
                    shared = sum(r[2] for r in ranges)
                    if shared >= args.min_gib * 2**30:
                        pairs.append((shared, fa, fb, ranges))
            pairs.sort(reverse=True)
            planned = 0
            for shared, fa, fb, ranges in pairs:
                planned += shared
                print(f"{shared / 2**30:8.2f} GiB  {os.path.basename(fb)}  <-  {os.path.basename(fa)}")
            print(f"planned shared bytes across {len(pairs)} pairs: {planned / 2**30:.2f} GiB "
                  f"{'(dry run)' if args.dry_run else ''}")
            if args.dry_run:
                return 0
            total_applied = 0
            notes = {}
            for shared, fa, fb, ranges in pairs:
                fd_a = os.open(fa, os.O_RDONLY)
                fd_b = os.open(fb, os.O_RDONLY)
                per_pair = 0
                try:
                    for off_a, off_b, ln in ranges:
                        mapped, note = fideduperange(fd_a, off_a, fd_b, off_b, ln)
                        per_pair += mapped
                        if note:
                            notes.setdefault(note, 0)
                            notes[note] += 1
                finally:
                    os.close(fd_a)
                    os.close(fd_b)
                total_applied += per_pair
                if int(per_pair / 2**30) or shared < per_pair:
                    print(f"applied {per_pair / 2**30:8.2f} GiB  {os.path.basename(fb)} <- "
                          f"{os.path.basename(fa)}")
            print(f"applied shared bytes total: {total_applied / 2**30:.2f} GiB")
            for note, cnt in sorted(notes.items()):
                print(f"  note: {note} x{cnt}")
        return 0

    if args.cmd == "share":
        return {"gen": _share_cmd_gen, "status": _share_cmd_status, "warm": _share_cmd_warm}[args.share_cmd](args)

    ap.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()