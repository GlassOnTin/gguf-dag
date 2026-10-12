// CUDA IPC chunk agent for gguf-dag store manifests.
//
//   agent publish <plan.txt> <pub.out>
//     Reads its chunk list from the plan, copies every chunk from the store
//     pack into device arenas, exports cudaIpcMemHandle_t per arena plus a
//     digest -> (arena, offset) map, and writes ARENA/MAP/READY lines to
//     pub.out.  Then waits for CONSUMER_DONE (or timeout), frees, exits.
//
//   agent consume <plan.txt> <pub.out> <con.out> [hold_seconds]
//     Waits for READY, opens every published arena via cudaIpcOpenMemHandle,
//     copies only chunks whose digests are NOT in the map into private
//     arenas, then verifies every chunk two ways:
//       host:  D2H copy + sha256 vs the plan
//       device: reduction kernel computing byte-sum / byte-xor vs the plan
//     (shared chunks are read through the attached IPC pages).  Reports
//     per-page VRAM deltas and timings, optionally holds memory for
//     hold_seconds so the wrapper can snapshot nvidia-smi, then exits.
//
//   agent selftest          sha256 sanity vs known vectors
//   agent b64 <0x-prefixed-hex>   encode arbitrary bytes (debug aid)

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorString(e_), __FILE__, __LINE__, #call);  \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

struct Chunk {
    std::string digest;  // 32 hex chars (blake2b-16)
    size_t pack_off;
    size_t size;
    std::string sha256;  // 64 hex
    unsigned long long sum;
    unsigned int xorb;
};

// ---------------------------------------------------------------- sha256 --
// Public-domain style, compact. Checked by `agent selftest`.
namespace sha256 {
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (blk[4 * i] << 24) | (blk[4 * i + 1] << 16) | (blk[4 * i + 2] << 8) | blk[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

static std::string hex_digest(const uint8_t* data, size_t n) {
    static const char* hx = "0123456789abcdef";
    uint32_t st[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[128];
    size_t i = 0;
    auto chunk = [&](const uint8_t* p) { compress(st, p); };
    for (; i + 64 <= n; i += 64) chunk(data + i);
    size_t rem = n - i;
    memcpy(buf, data + i, rem);
    buf[rem] = 0x80;
    size_t padded = (rem + 1 <= 56) ? 64 : 128;
    memset(buf + rem + 1, 0, padded - rem - 1);
    uint64_t bits = (uint64_t)n * 8;
    for (int k = 0; k < 8; k++) buf[padded - 1 - k] = (bits >> (8 * k)) & 0xff;
    chunk(buf);
    if (padded == 128) chunk(buf + 64);
    std::string out;
    for (int k = 0; k < 8; k++)
        for (int shift = 28; shift >= 0; shift -= 4) out += hx[(st[k] >> shift) & 0xf];
    return out;
}
}  // namespace sha256

// ------------------------------------------------------------------ base64 --
static const char* B64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string b64encode(const uint8_t* p, size_t n) {
    std::string out;
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
        out += B64[(v >> 18) & 63];
        out += B64[(v >> 12) & 63];
        out += B64[(v >> 6) & 63];
        out += B64[v & 63];
    }
    if (i + 1 == n) {
        uint32_t v = p[i] << 16;
        out += B64[(v >> 18) & 63];
        out += B64[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == n) {
        uint32_t v = (p[i] << 16) | (p[i + 1] << 8);
        out += B64[(v >> 18) & 63];
        out += B64[(v >> 12) & 63];
        out += B64[(v >> 6) & 63];
        out += "=";
    }
    return out;
}
static std::vector<uint8_t> b64decode(const std::string& s) {
    auto val = [](char c) -> int {
        const char* q = strchr(B64, c);
        return q ? int(q - B64) : 0;
    };
    std::vector<uint8_t> out;
    int acc = 0, bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == ' ') continue;
        acc = (acc << 6) | val(c);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t((acc >> bits) & 0xff));
        }
    }
    return out;
}

// ------------------------------------------------------------------ kernel --
__global__ void chunk_fingerprint(const unsigned char* p, size_t n,
                                  unsigned long long* sum_out,
                                  unsigned int* xor_out) {
    __shared__ unsigned long long sdata[256];
    __shared__ unsigned int xdata[256];
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = 0;
    unsigned int xacc = 0;
    for (; i < n; i += gridDim.x * blockDim.x) {
        acc += p[i];
        xacc ^= (unsigned)p[i];
    }
    sdata[threadIdx.x] = acc;
    xdata[threadIdx.x] = xacc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            xdata[threadIdx.x] ^= xdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // d_sum/d_xor are zero-seeded, so every block's partial must land here
        atomicAdd(sum_out, sdata[0]);
        atomicXor(xor_out, xdata[0]);
    }
}

// Device-side fingerprint of one chunk: seeded zero atomics, grid of blocks.
static void device_fingerprint(const unsigned char* dev_ptr, size_t n,
                               unsigned long long* d_sum, unsigned int* d_xor,
                               unsigned long long* h_sum, unsigned int* h_xor) {
    CK(cudaMemsetAsync(d_sum, 0, 8));
    CK(cudaMemsetAsync(d_xor, 0, 4));
    int blocks = 4;
    chunk_fingerprint<<<blocks, 256>>>(dev_ptr, n, d_sum, d_xor);
    CK(cudaGetLastError());
    CK(cudaMemcpy(h_sum, d_sum, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_xor, d_xor, 4, cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------- plan ---
struct Plan {
    std::string pack_path;
    size_t arena_bytes = 0;
    std::vector<Chunk> publish;
    std::vector<Chunk> consume;
};

static Plan load_plan(const char* path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "cannot open plan %s\n", path);
        std::exit(2);
    }
    Plan p;
    std::string line, role;
    auto handle = [&](const std::string& line) {
        if (line.rfind("PACK ", 0) == 0) {
            p.pack_path = line.substr(5);
        } else if (line.rfind("ARENA_BYTES ", 0) == 0) {
            p.arena_bytes = strtoull(line.c_str() + 12, nullptr, 10);
        } else if (line.rfind("PUBLISH_END", 0) == 0 ||
                   line.rfind("CONSUME_END", 0) == 0) {
            role.clear();
        } else if (line.rfind("PUBLISH_CHUNK ", 0) == 0) {
            role = "PUBLISH";
        } else if (line.rfind("CONSUME_CHUNK ", 0) == 0) {
            role = "CONSUME";
        }
        if (role.empty()) return;
        // role is PUBLISH or CONSUME from the current line's prefix
        Chunk c;
        const char* s = line.c_str();
        char dig[64], sha[128];
        unsigned long long off_, sz_, sum_;
        unsigned int xor_;
        if (sscanf(s, "%*s %63s %llu %llu %127s %llu %u", dig, &off_, &sz_, sha,
                   &sum_, &xor_) != 6) {
            std::fprintf(stderr, "bad plan line: %.40s\n", s);
            std::exit(2);
        }
        c.digest = dig;
        c.pack_off = off_;
        c.size = sz_;
        c.sha256 = sha;
        c.sum = sum_;
        c.xorb = xor_;
        (role == "PUBLISH" ? p.publish : p.consume).push_back(c);
    };
    while (std::getline(f, line)) handle(line);
    return p;
}

// ------------------------------------------------------------ placement ----
// Two-phase: lay out chunk -> (arena, offset) arithmetically, then allocate
// arenas at exact byte counts, then copy.  Avoids a padded final arena.
struct Arena { void* dev = nullptr; size_t used = 0; size_t bytes = 0; };

struct Placer {
    size_t arena_bytes;
    std::vector<size_t> lens;              // per-arena byte totals
    std::unordered_map<std::string, std::pair<int, size_t>> map;
    explicit Placer(size_t ab) : arena_bytes(ab) {}
    void add(const Chunk& c) {
        size_t off = lens.empty() ? 0 : lens.back();
        if (lens.empty() || off + c.size > arena_bytes) {
            lens.push_back(0);
            off = 0;
        }
        map[c.digest] = {int(lens.size()) - 1, off};
        lens.back() = off + c.size;
    }
    size_t alloc() {                        // exact-size cudaMalloc per arena
        for (size_t L : lens) {
            void* d = nullptr;
            CK(cudaMalloc(&d, L));
            arenas_.push_back({d, L, L});
        }
        return bytes();
    }
    size_t bytes() const {
        size_t s = 0;
        for (size_t L : lens) s += L;
        return s;
    }
    std::vector<Arena> arenas_;
    size_t total_bytes() const { return bytes(); }
};

// -------------------------------------------------------------- mmap pack ---
class PackMap {
    int fd_ = -1;
    void* mm_ = nullptr;
    size_t len_ = 0;

   public:
    ~PackMap() { close(); }
    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { std::perror("open pack"); std::exit(2); }
        len_ = (size_t)lseek64_(fd_);
        mm_ = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mm_ == MAP_FAILED) { std::perror("mmap pack"); std::exit(2); }
    }
    void* at(size_t off) const { return (char*)mm_ + off; }
    static off_t lseek64_(int fd) {
        // fstat is enough: pack length == end offset
        struct ::stat st{};
        if (fstat(fd, &st) || !st.st_size) { std::fprintf(stderr, "fstat\n"); std::exit(2); }
        return st.st_size;
    }
    void close() {
        if (mm_ && mm_ != MAP_FAILED) munmap(mm_, len_);
        if (fd_ >= 0) ::close(fd_);
        mm_ = nullptr;
        fd_ = -1;
    }
};

static std::string file_tail(const std::string& path, long max_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    std::streamoff start = sz > max_bytes ? sz - max_bytes : 0;
    f.seekg(start);
    std::string data(size_t(sz - start), '\0');
    f.read(data.data(), (std::streamsize)data.size());
    return data;
}

// ------------------------------------------------------------------ modes ---
static int mode_selftest() {
    const uint8_t abc[] = {'a', 'b', 'c'};
    std::string h1 = sha256::hex_digest(abc, 3);  // ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    std::vector<uint8_t> big(1000, 0x61);
    std::string h2 = sha256::hex_digest(big.data(), big.size());
    std::printf("sha256_abc %s\nsha256_1000a %s\nipchandle_size %zu\n",
                h1.c_str(), h2.c_str(), sizeof(cudaIpcMemHandle_t));
    return 0;
}

static volatile sig_atomic_t g_stop = 0;
static void on_term(int) { g_stop = 1; }

static int mode_publish(const char* plan_path, const char* out_path,
                        const char* con_path, double timeout_s) {
    Plan plan = load_plan(plan_path);
    PackMap pack;
    pack.open(plan.pack_path);

    std::signal(SIGTERM, on_term);
    std::signal(SIGINT, on_term);

    CK(cudaFree(0));
    size_t free0 = 0, total = 0;
    CK(cudaMemGetInfo(&free0, &total));
    int dev_i = -1;
    char pci[32] = "?";
    CK(cudaGetDevice(&dev_i));
    cudaDeviceGetPCIBusId(pci, sizeof(pci), dev_i);
    std::fprintf(stderr, "[pub] device %d pci %s free %.0f MiB\n", dev_i, pci,
                 free0 / 1048576.0);

    auto t0 = std::chrono::steady_clock::now();
    Placer placer(plan.arena_bytes);
    for (const Chunk& c : plan.publish) placer.add(c);
    placer.alloc();
    // one bulk pass: arena contents arrive in slot order; copy chunk by chunk
    std::ofstream out(out_path);
    double copy_s = 0;
    auto tc0 = std::chrono::steady_clock::now();
    for (const Chunk& c : plan.publish) {
        auto [aid, off] = placer.map[c.digest];
        CK(cudaMemcpy((char*)placer.arenas_[aid].dev + off, pack.at(c.pack_off),
                      c.size, cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());
    copy_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();

    cudaIpcMemHandle_t h;
    for (size_t i = 0; i < placer.arenas_.size(); i++) {
        CK(cudaIpcGetMemHandle(&h, placer.arenas_[i].dev));
        uint8_t raw[64];
        memcpy(raw, &h, sizeof(h));
        out << "ARENA " << i << " " << placer.arenas_[i].bytes << " "
            << b64encode(raw, sizeof(h)) << "\n";
    }
    for (const Chunk& c : plan.publish) {
        auto [aid, off] = placer.map[c.digest];
        out << "MAP " << c.digest << " " << aid << " " << off << "\n";
    }
    size_t free1 = 0;
    CK(cudaMemGetInfo(&free1, &total));
    out << "PUB_STATS chunks_total " << plan.publish.size() << "\n"
        << "PUB_STATS bytes_resident " << placer.total_bytes() << "\n"
        << "PUB_STATS arenas " << placer.arenas_.size() << "\n"
        << "PUB_STATS copy_s " << copy_s << "\n"
        << "PUB_STATS mem_used " << (free0 - free1) << "\n"
        << "READY\n" << std::flush;
    std::fprintf(stderr, "[pub] %zu chunks (%zu B) in %zu arenas, copy %.2fs\n",
                 plan.publish.size(), placer.total_bytes(), placer.arenas_.size(),
                 copy_s);

    auto t_start = std::chrono::steady_clock::now();
    while (!g_stop) {
        std::string tail = file_tail(con_path, 4096);
        if (tail.find("CONSUMER_DONE") != std::string::npos) break;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                .count() > timeout_s)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    for (auto& a : placer.arenas_) if (a.dev) CK(cudaFree(a.dev));
    pack.close();
    auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[pub] done, wall %.2fs\n", wall);
    return 0;
}

static int mode_consume(const char* plan_path, const char* pub_path,
                        const char* con_out_path, double pub_timeout_s,
                        double hold_s) {
    Plan plan = load_plan(plan_path);
    PackMap pack;
    pack.open(plan.pack_path);

    // wait for publisher READY
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        if (file_tail(pub_path, 1 << 16).find("READY\n") != std::string::npos) break;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count() > pub_timeout_s) {
            std::fprintf(stderr, "publisher did not signal READY in %.0fs\n", pub_timeout_s);
            return 3;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    double wait_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // parse arenas + map from pub.out
    std::ifstream pubf(pub_path);
    std::string line;
    std::unordered_map<int, std::string> handles;   // arena -> b64
    std::vector<size_t> arena_bytes;
    std::unordered_map<std::string, std::pair<int, size_t>> pmap;
    while (std::getline(pubf, line)) {
        if (line.rfind("ARENA ", 0) == 0) {
            int idx; unsigned long long bytes; char b64[256];
            if (sscanf(line.c_str(), "ARENA %d %llu %255s", &idx, &bytes, b64) != 3)
                continue;
            handles[idx] = b64;
            if (arena_bytes.size() <= (size_t)idx) arena_bytes.resize(idx + 1, 0);
            arena_bytes[idx] = bytes;
        } else if (line.rfind("MAP ", 0) == 0) {
            char dig[64]; int aid; unsigned long long off;
            if (sscanf(line.c_str(), "MAP %63s %d %llu", dig, &aid, &off) == 3)
                pmap[dig] = {aid, (size_t)off};
        }
    }

    CK(cudaFree(0));
    size_t free0 = 0, total = 0;
    CK(cudaMemGetInfo(&free0, &total));

    // split consume chunks
    size_t shared_bytes = 0, unique_bytes = 0;
    for (const Chunk& c : plan.consume)
        (pmap.count(c.digest) ? shared_bytes : unique_bytes) += c.size;

    // private arenas for unique chunks
    auto t_copy0 = std::chrono::steady_clock::now();
    Placer priv(plan.arena_bytes);
    for (const Chunk& c : plan.consume)
        if (!pmap.count(c.digest)) priv.add(c);
    priv.alloc();
    for (const Chunk& c : plan.consume) {
        if (pmap.count(c.digest)) continue;  // shared: attached, not copied
        auto [aid, off] = priv.map[c.digest];
        CK(cudaMemcpy((char*)priv.arenas_[aid].dev + off, pack.at(c.pack_off),
                      c.size, cudaMemcpyHostToDevice));
    }
    double copy_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_copy0).count();

    // attach publisher arenas via IPC
    auto t_att0 = std::chrono::steady_clock::now();
    std::vector<void*> attached(arena_bytes.size(), nullptr);
    for (auto& [idx, b64] : handles) {
        auto raw = b64decode(b64);
        if (raw.size() != sizeof(cudaIpcMemHandle_t)) {
            std::fprintf(stderr, "handle %d has %zu bytes, expected %zu\n", idx,
                         raw.size(), sizeof(cudaIpcMemHandle_t));
            return 3;
        }
        cudaIpcMemHandle_t h{};
        memcpy(&h, raw.data(), sizeof(h));
        // same-device open still requires the lazy-peer-access flag;
        // with flags 0 the driver returns cudaErrorInvalidValue
        CK(cudaIpcOpenMemHandle(&attached[idx], h, cudaIpcMemLazyEnablePeerAccess));
    }
    double attach_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_att0).count();

    // device workspace for fingerprints
    unsigned long long *d_sum; unsigned int *d_xor;
    CK(cudaMalloc(&d_sum, 8));
    CK(cudaMalloc(&d_xor, 4));
    unsigned long long h_sum; unsigned int h_xor;

    // verification
    auto t_ver0 = std::chrono::steady_clock::now();
    size_t ok_sha = 0, ok_dev = 0, fails = 0;
    std::vector<uint8_t> host_buf;
    host_buf.reserve(1 << 20);
    for (const Chunk& c : plan.consume) {
        void* dev_ptr;
        auto it = pmap.find(c.digest);
        if (it != pmap.end()) {
            dev_ptr = (char*)attached[it->second.first] + it->second.second;
        } else {
            auto [aid, off] = priv.map[c.digest];
            dev_ptr = (char*)priv.arenas_[aid].dev + off;
        }
        host_buf.resize(c.size);
        CK(cudaMemcpy(host_buf.data(), dev_ptr, c.size, cudaMemcpyDeviceToHost));
        if (sha256::hex_digest(host_buf.data(), c.size) == c.sha256) ok_sha++; else fails++;
        device_fingerprint((const unsigned char*)dev_ptr, c.size, d_sum, d_xor, &h_sum, &h_xor);
        if (h_sum == c.sum && h_xor == c.xorb) ok_dev++; else fails++;
    }
    double verify_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ver0).count();

    // dummy footprint so the consumer shows in nvidia-smi even when fully shared
    void* marker = nullptr;
    CK(cudaMalloc(&marker, 64u << 20));
    CK(cudaMemset(marker, 1, 64u << 20));

    size_t free1 = 0;
    CK(cudaMemGetInfo(&free1, &total));
    double own_used = free0 - free1;
    int dev_i = -1;
    char pci[32] = "?";
    CK(cudaGetDevice(&dev_i));
    cudaDeviceGetPCIBusId(pci, sizeof(pci), dev_i);
    std::fprintf(stderr, "[con] device %d pci %s\n", dev_i, pci);
    size_t attached_bytes = 0;
    for (auto& [idx, b64] : handles) attached_bytes += arena_bytes[idx];

    // HOLDING lets the wrapper snapshot nvidia-smi with both agents resident;
    // CONSUMER_DONE only lands after hold so the publisher cannot free early
    std::ofstream of(con_out_path);
    of << "CON_STATS shared_bytes " << shared_bytes << "\n"
       << "CON_STATS unique_bytes " << unique_bytes << "\n"
       << "CON_STATS attached_bytes_total " << attached_bytes << "\n"
       << "CON_STATS mem_used_self " << own_used << "\n"
       << "CON_STATS wait_s " << wait_s << "\n"
       << "CON_STATS copy_unique_s " << copy_s << "\n"
       << "CON_STATS attach_s " << attach_s << "\n"
       << "CON_STATS verify_s " << verify_s << "\n"
       << "CON_STATS sha_ok " << ok_sha << " dev_ok " << ok_dev << " fails " << fails << "\n"
       << std::flush;
    of << "HOLDING\n" << std::flush;
    of.close();
    std::fprintf(stderr,
                 "[con] shared %.2f GiB attached, unique %.2f GiB copied in %.2fs; "
                 "attach %.3fs; verify %zu/%zu sha + dev, fails %zu, %.2fs; own VRAM %.0f MiB\n",
                 shared_bytes / 1073741824.0, unique_bytes / 1073741824.0, copy_s,
                 attach_s, ok_sha, plan.consume.size(), fails, verify_s, own_used / 1048576.0);

    if (hold_s > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
    }
    std::ofstream done(con_out_path, std::ios::app);
    done << "CONSUMER_DONE\n";
    done.close();
    CK(cudaFree(marker));
    for (auto& a : priv.arenas_) if (a.dev) CK(cudaFree(a.dev));
    for (auto* p : attached) if (p) CK(cudaIpcCloseMemHandle(p));
    CK(cudaFree(d_sum));
    CK(cudaFree(d_xor));
    pack.close();
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "selftest") return mode_selftest();
    if (argc >= 5 && std::string(argv[1]) == "publish")
        return mode_publish(argv[2], argv[3], argv[4], argc > 5 ? atof(argv[5]) : 900.0);
    if (argc >= 6 && std::string(argv[1]) == "consume")
        return mode_consume(argv[2], argv[3], argv[4], 900.0, argc > 6 ? atof(argv[6]) : 0.0);
    std::fprintf(stderr,
                 "usage:\n  %s publish <plan.txt> <pub.out> <con.out> [timeout_s]\n"
                 "  %s consume <plan.txt> <pub.out> <con.out> [hold_s]\n"
                 "  %s selftest\n",
                 argv[0], argv[0], argv[0]);
    return 1;
}