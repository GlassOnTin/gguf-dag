"""Orchestrate one CUDA IPC chunk-sharing demo and assert its numbers.

Stages: pub agent (copies its manifest's chunks from the pack into arenas,
exports handles) -> con agent (attaches shared pages, copies only unique
chunks, verifies every chunk two ways) -> snapshot nvidia-smi while both hold
memory -> assert verification and VRAM accounting.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nvidia_mem(retries=6, delay=0.5):
    """pid -> MiB of resident compute apps.

    A nvidia-smi query can come back empty on returncode 0 while other NVML
    clients are hammering the driver concurrently (observed with a 1 Hz
    sampler running alongside) — retry on empty and log the raw result if it
    won't fill.
    """
    r = None
    for attempt in range(retries):
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
            capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            mem = {}
            for line in r.stdout.strip().splitlines():
                if "," not in line:
                    continue
                pid, mib = line.split(",")
                if mib.strip().split()[0] == "N/A":
                    continue
                mem[int(pid)] = int(mib.strip().split()[0])
            if mem:
                return mem
        if attempt + 1 < retries:
            time.sleep(delay)
    if retries:
        print(f"[debug] nvidia-smi query-compute-apps stayed empty after {retries} tries: "
              f"rc={getattr(r, 'returncode', None)} stdout={getattr(r, 'stdout', '')[-200:]!r} "
              f"stderr={getattr(r, 'stderr', '')[-200:]!r}")
    return {}


def wait_for(path, marker, timeout, poll=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            if marker in path.read_text(errors="replace"):
                return time.time() - t0
        time.sleep(poll)
    raise TimeoutError(f"{marker!r} not seen in {path} within {timeout}s")


def parse_section(path, tag):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(tag):
            parts = line.split()
            for i in range(1, len(parts) - 1, 2):
                try:
                    out[parts[i]] = float(parts[i + 1])
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/tmp/ggdag-cli")
    ap.add_argument("--pub", required=True, help="manifest id to publish")
    ap.add_argument("--con", required=True, help="manifest id to consume")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--workdir", default="/tmp/ggipc")
    ap.add_argument("--hold", type=float, default=8.0)
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(exist_ok=True)
    plan, pub_out, con_out = wd / "plan.txt", wd / "pub.out", wd / "con.out"
    for p in (plan, pub_out, con_out):
        p.unlink(missing_ok=True)

    print("== build plan (hashes via store pack) ==")
    t0 = time.time()
    subprocess.run([sys.executable, str(HERE / "plan.py"), "--store", args.store,
                    "--pub", args.pub, "--con", args.con, "--out", str(plan)],
                   check=True)
    print(f"plan built in {time.time() - t0:.1f}s")

    before = nvidia_mem(retries=0)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)

    print("== launch publisher ==")
    pub_err = open(wd / "pub.err", "w")
    pub = subprocess.Popen(
        [str(HERE / "agent"), "publish", str(plan), str(pub_out), str(con_out)],
        env=env, stderr=pub_err, text=True)
    con = None
    con_err = None
    try:
        try:
            wait_for(pub_out, "READY", 300)
        except TimeoutError:
            err = (wd / "pub.err").read_text()
            print(f"publisher timed out; stderr:\n{err[-2000:]}")
            out = pub_out.read_text() if pub_out.exists() else ""
            print(f"pub.out had {len(out)} bytes, ends: ...{out[-200:]}")
            raise

        print("== launch consumer (holds VRAM for snapshot) ==")
        con_err = open(wd / "con.err", "w")
        con = subprocess.Popen(
            [str(HERE / "agent"), "consume", str(plan), str(pub_out), str(con_out),
             str(args.hold)],
            env=env, stderr=con_err, text=True)
        # Sample the GPU total while con is active (attach/copy/verify are still
        # running).  NVML is consistent during activity; once agents idle at
        # HOLDING its totals collapse to ~3 MiB while per-pid ctx rows persist —
        # a driver accounting artifact, reproduced across runs, so we do not
        # assert on idle-phase NVML.
        gpu_idx = int(args.gpu)
        gpu_max = 0
        for _ in range(12):
            gpus_used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip().splitlines()
            if len(gpus_used) > gpu_idx:
                gpu_max = max(gpu_max, int(gpus_used[gpu_idx].split()[0]))
            time.sleep(0.7)
            if con_out.exists() and "HOLDING" in con_out.read_text(errors="replace"):
                break
        holding = nvidia_mem()
        rc_con = con.wait(timeout=180)
        wait_for(con_out, "CONSUMER_DONE", 30)
    finally:
        # A dying consumer (OOM at its context init, a wait timeout, Ctrl-C)
        # must never leak the publisher at full residency — every error path
        # reaches here, so both agents are reaped and the handles closed.
        for proc in (con, pub):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        pub_err.close()
        if con_err is not None:
            con_err.close()

    pub_s = parse_section(pub_out, "PUB_STATS")
    con_s = parse_section(con_out, "CON_STATS")
    after = nvidia_mem(retries=0)

    pub_bytes = pub_s.get("bytes_resident", 0)
    pub_own = pub_s.get("mem_used", 0)      # in-process cudaMemGetInfo delta
    shared = con_s.get("shared_bytes", 0)
    unique = con_s.get("unique_bytes", 0)
    con_own = con_s.get("mem_used_self", 2**64)
    n_chunks = con_s.get("sha_ok", -1)
    fails = con_s.get("fails", -1)
    pub_pid_mem = holding.get(pub.pid, 0)
    con_pid_mem = holding.get(con.pid, 0)

    print(f"[agent@HOLDING] nvidia-smi per-pid rows (best-effort telemetry): "
          f"pub {pub_pid_mem} MiB, con {con_pid_mem} MiB")
    print(f"pub: {pub_s.get('chunks_total'):.0f} chunks, {pub_bytes / 2**20:.0f} MiB "
          f"in {pub_s.get('arenas'):.0f} arenas, copy {pub_s.get('copy_s'):.2f}s, "
          f"own VRAM {pub_own / 2**20:.0f} MiB")
    print(f"con: shared {shared / 2**20:.0f} MiB attached, unique {unique / 2**20:.0f} MiB "
          f"copied in {con_s.get('copy_unique_s'):.2f}s, attach {con_s.get('attach_s'):.3f}s, "
          f"verify {con_s.get('verify_s'):.2f}s, own VRAM {con_own / 2**20:.0f} MiB")
    print(f"verify: sha_ok={n_chunks:.0f} fails={fails:.0f}")
    print(f"GPU{gpu_idx} total (peak over active verify window): {gpu_max} MiB")

    # consumer's own footprint must be unique chunks + marker + ctx slack
    ctx = max(0.0, pub_own - pub_bytes + 100 * 2**20)
    naive = (pub_own + con_own + shared + 2 * 500 * 2**20) / 2**20  # + ~0.5 GiB ctx each
    ok = fails == 0 and n_chunks >= 0 and rc_con == 0
    ok = ok and con_own < unique + 64 * 2**20 + ctx
    if gpu_max > 0:
        print(f"GPU accounting: peak {gpu_max} MiB during verify; two independent "
              f"copies would be ~{naive:.0f} MiB")
    print(f"con own-VRAM bound: {con_own / 2**20:.0f} MiB < unique {unique / 2**20:.0f} "
          f"MiB + marker 64 + ctx slack {ctx / 2**20:.0f} MiB")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())