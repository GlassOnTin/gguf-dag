# gguf-dag

Splits GGUF model files into a content-addressed store of fixed-size chunks plus
per-file manifests, and rebuilds files from it **byte-identical**. The
round-trip is the acceptance test.

## Why

Local model collections hold mostly-identical bytes: shard groups of one model,
the same model re-quantized, fine-tunes of the same base. Each GGUF file repeats
all of it. I measured a local corpus of 14 files, 5,488 tensors, 488 GiB:

- 379 GiB unique after 64 KiB chunking, so 109 GiB was cross-file duplication
- shard groups (Q4_K_XL split into 4 shards) recover 98.3% of each other's chunks
- a Q4_K_M vs Q8_0 pair of the same model shares 21.9% (the recipe-matched
  fraction); a fine-tune vs its base shares 0.3%, because a 2.2% RMS weight
  drift swamps the 0.2% re-encoding noise of a shared quant recipe
- the bookkeeping itself is thin: 32-byte index records per unique chunk,
  0.025% of pack bytes

The manifest is a chain of hashes (header blob → per-tensor chunk-digest
streams → pack chunks), which doubles as tamper detection: `verify` recomputes
the chain and names the first problem it finds.

`gguf-dag` does storage only. It does not compress (GGML k-quants already sit
at code entropy; zstd-19 recovers 6–12% from them and 23–93% only from float
payloads), and it does not change how llama.cpp loads anything.

## Usage

```sh
pip install .            # or: PYTHONPATH=src python3 -m gguf_dag.cli ...
gguf-dag --store ./store add /models/qwen3-shard-00001.gguf *.gguf
gguf-dag --store ./store ls <id-prefix>
gguf-dag --store ./store verify all
gguf-dag --store ./store rebuild <id-prefix> -o rebuilt.gguf
gguf-dag --store ./store stats
```

Byte-identity check after a rebuild:

```sh
gguf-dag sha256 rebuilt.gguf && sha256sum /models/original.gguf
```

Manifests can be HMAC-signed (`--hmac-key keyfile`, HMAC-BLAKE2b over the
canonical JSON) so tampering is detected even when chunk hashes are replaced
consistently.

## Store layout

```
store/
  chunks.pack     append-only concat of unique chunks, first-seen order
  chunks.idx      sorted 32 B records: blake2b-16 digest, u64 offset, u32 len
  manifests/<id>/ manifest.json (small), refs.bin (per-tensor digest streams),
                  header.bin (zlib header region)
  lock            flock — one writer at a time
```

An `id` is blake2b-16 of the file's header region; a store holding manifests
for many models is just these directories. Chunks are anchored at tensor
boundaries (a chunk never straddles two tensors), so a future consumer can map
a chunk digest back to a named tensor.

The parser needs no GGML size table: tensor *N*'s span runs from its offset to
tensor *N+1*'s offset (last one to EOF), so padding and new quant types need no
code change. Test files with junk between tensors and after the last tensor
round-trip byte-identically, and the spans agree with `gguf-py`'s
`t.n_bytes` on every tensor of three real files.

## Designed for (not implemented here)

- A reloadable llama.cpp loader hook: before mmap'ing, ask a local daemon
  "which chunk digests are already resident?", then build addresses from a
  manifest instead of re-reading. Cost stays in the load phase; llama.cpp
  dequantizes quantized blocks in registers/L2 anyway, so chunk-level
  indirection never touches the compute path.

## CUDA IPC prototype

`ipc/` (not installed by `pip`, build with `make -C ipc`, needs nvcc) is a
working proof of the cross-instance VRAM sharing designed around this store.
A `plan.py` pass reads two manifests from a store and writes a chunk plan:
which chunks the publisher places in device arenas and which digests the
consumer can expect to find already resident. `agent.cu` runs the two sides:

- **publish** — copies listed chunks from `chunks.pack` (mmap'd) into
  exact-size device arenas, exports one `cudaIpcMemHandle_t` per arena plus a
  digest → (arena, offset) map, waits for the consumer.
- **consume** — opens the published arenas via IPC, privately copies only
  chunks it cannot attach, then verifies *every* chunk two ways: host sha256
  of a D2H read, and a device-side reduction (byte sum + byte xor) against
  fingerprints that came with the plan. Shared chunks are verified through
  the attached pages.

Measured on 2× 32 GiB RTX 5090, driver 595.84, chunks from the demo store:

| case | publisher | consumer attach | consumer own VRAM | verify |
|---|---|---|---|---|
| same file twice | 2490 MiB, 3 arenas, copy 1.3 s | 2490 MiB, 0.014 s | **70 MiB** | 39865/39865 sha + device, fails 0 |
| Q8_0 ↔ Q4_K_M | 3935 MiB, 4 arenas, copy 2.1 s | 860 MiB shared | 1706 MiB (1630 MiB unique copied, 0.84 s) | 39865/39865, fails 0 |

The consumer's own VRAM bound is read in-process via `cudaMemGetInfo` deltas,
which is the honest number: a second instance of a fully-shared model holds
its context plus one 64 MiB marker, and a Q4 consumer next to its Q8 sibling
holds 69% of its own bytes instead of 100%. `demo.py` asserts those bounds and
the two-way verification.

One driver caveat found while measuring: `nvidia-smi` totals are accurate
while these processes run (`utilization > 0`) and collapse once they sit idle
holding allocated memory (3–500 MiB reported for 3.5 GiB actually resident,
per-pid and per-GPU totals disagreeing with each other). Reproduced across
runs on 595.84; the demo therefore samples peak totals during the active
verify window and treats idle-phase NVML output as telemetry, not evidence.

## Limits

- Single writer (flock); concurrent readers are fine.
- The span logic needs strictly increasing tensor offsets, which the GGUF
  spec implies and all files I've seen satisfy; parser rejects otherwise.
- Fixed-size chunks: no content-defined chunking, so offsets that shift past a
  chunk boundary (re-quantization, layout changes) defeat sharing. Deterministic
  re-quantization that reproduces byte-identical blocks (as merged shard
  groups did here, 98.3%) is detected.
- GGUF v2/v3 only; no nested-array KV parsing.
- Corpus tested: BF16, Q4_K_XL, Q4_K_M, Q8_0, MTP, mmproj files. No Q2-tier or
  IQ-tier files exercised (none in my corpus).

## License

AGPL-3.0