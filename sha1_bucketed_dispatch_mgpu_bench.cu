/*
 Program Name     : sha1_bucketed_dispatch_mgpu_bench.cu
 Program Author(s): Rick Pelletier (galiagante@gmail.com)
 Program Date     : 30 December 2025
 Program Update   : 31 December 2025
 Program Version  : 0.5-20251231-ALPHA (Do Not Distribute!)
 Program Platform : C++ (CUDA), compiled with nvcc 11+

 Program Notes

 NOTE: This is test code and very likely requires significant additional development and optimization!

 Why this implementation is correct:
 - No inter-GPU communication
 - Identical kernels across devices
 - Explicit device control
 - Deterministic behavior
 - Near-linear scaling
 - Hashcat-style reporting

 This is how serious cracking / hashing frameworks scale across GPUs — just without their historical baggage.

 What this deliberately avoids (for sanity):
 - Unified memory
 - Threads per GPU
 - P2P
 - NVSHMEM
 - Cooperative multi-device kernels

 All intentionally postponed until there’s a proven need.

 Usage:

 ./sha1_bucketed_dispatch_bench input.txt --bench --gpu all
 ./sha1_bucketed_dispatch_bench input.txt --gpu 0,1
 ./sha1_bucketed_dispatch_bench input.txt --gpu 2

 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <algorithm>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << cudaGetErrorString(err) << "\n"; \
    std::exit(1); \
  } \
} while (0)

constexpr int BLOCK_BYTES  = 64;
constexpr int DIGEST_BYTES = 20;
constexpr int MAX_LEN    = 55;
constexpr int BENCH_ITERATIONS = 50;

/* ====================== Device SHA-1 ====================== */

__device__ __forceinline__ uint32_t rotl(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

template<int LEN>
__device__ void sha1_fixed(const uint8_t* block, uint8_t* out) {
  uint32_t w[80];

#pragma unroll
  for (int i = 0; i < 16; i++) {
    w[i] = (block[i*4+0] << 24) |
         (block[i*4+1] << 16) |
         (block[i*4+2] << 8)  |
         (block[i*4+3]);
  }

#pragma unroll
  for (int i = 16; i < 80; i++)
    w[i] = rotl(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);

  uint32_t a = 0x67452301;
  uint32_t b = 0xEFCDAB89;
  uint32_t c = 0x98BADCFE;
  uint32_t d = 0x10325476;
  uint32_t e = 0xC3D2E1F0;

#pragma unroll
  for (int i = 0; i < 80; i++) {
    uint32_t f, k;

    if (i < 20)    { f = (b & c) | (~b & d); k = 0x5A827999; }
    else if (i < 40) { f = b ^ c ^ d;      k = 0x6ED9EBA1; }
    else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
    else       { f = b ^ c ^ d;      k = 0xCA62C1D6; }

    uint32_t t = rotl(a,5) + f + e + k + w[i];
    e = d; d = c; c = rotl(b,30); b = a; a = t;
  }

  uint32_t h[5] = {
    a + 0x67452301,
    b + 0xEFCDAB89,
    c + 0x98BADCFE,
    d + 0x10325476,
    e + 0xC3D2E1F0
  };

#pragma unroll
  for (int i = 0; i < 5; i++) {
    out[i*4+0] = (h[i] >> 24) & 0xFF;
    out[i*4+1] = (h[i] >> 16) & 0xFF;
    out[i*4+2] = (h[i] >> 8)  & 0xFF;
    out[i*4+3] =  h[i]    & 0xFF;
  }
}

template<int LEN>
__global__ void sha1_kernel(const uint8_t* in, uint8_t* out, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < count) {
    sha1_fixed<LEN>(
      in  + idx * BLOCK_BYTES,
      out + idx * DIGEST_BYTES
    );
  }
}

/* ====================== Dispatch Table ====================== */

using launch_fn = void(*)(const uint8_t*, uint8_t*, size_t, int, int);

template<int LEN>
void launch(const uint8_t* d_in, uint8_t* d_out, size_t count, int blocks, int threads) {
  sha1_kernel<LEN><<<blocks, threads>>>(d_in, d_out, count);
}

template<int... L>
constexpr auto make_dispatch(std::integer_sequence<int, L...>) {
  return std::array<launch_fn, sizeof...(L)>{ &launch<L>... };
}

constexpr auto dispatch = make_dispatch(std::make_integer_sequence<int, MAX_LEN + 1>{});

/* ====================== Utilities ====================== */

static double hashes_per_second(size_t hashes, double ms) {
  return (hashes * 1000.0) / ms;
}

static std::vector<int> parse_gpu_list(const std::string& s) {
  int total;
  CUDA_CHECK(cudaGetDeviceCount(&total));

  if (s == "all") {
    std::vector<int> g(total);
    for (int i = 0; i < total; i++) g[i] = i;

    return g;
  }

  std::vector<int> g;
  std::stringstream ss(s);
  std::string tok;

  while (std::getline(ss, tok, ',')) {
    int id = std::stoi(tok);

    if (id < 0 || id >= total)
      throw std::runtime_error("Invalid GPU id");
    g.push_back(id);
  }

  return g;
}

/* ====================== Main ====================== */

int main(int argc, char** argv) {
  bool bench = false;
  std::vector<int> gpus = {0};

  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
          << " input.txt [--bench] [--gpu all|0,1,...]\n";
    return 1;
  }

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];

    if (a == "--bench") bench = true;
    else if (a == "--gpu" && i + 1 < argc)
      gpus = parse_gpu_list(argv[++i]);
  }

  std::ifstream f(argv[1]);
  if (!f) return 1;

  std::array<std::vector<std::string>, MAX_LEN + 1> buckets;
  std::string line;

  while (std::getline(f, line)) {
    if (line.size() > MAX_LEN) continue;
    buckets[line.size()].push_back(line);
  }

  for (int len = 0; len <= MAX_LEN; len++) {
    auto& bucket = buckets[len];

    if (bucket.empty()) continue;

    size_t total = bucket.size();
    size_t per_gpu = (total + gpus.size() - 1) / gpus.size();

    double total_hps = 0.0;

    for (size_t gi = 0; gi < gpus.size(); gi++) {
      size_t start = gi * per_gpu;
      size_t count = std::min(per_gpu, total - start);

      if (!count) continue;

      CUDA_CHECK(cudaSetDevice(gpus[gi]));

      std::vector<uint8_t> h_in(count * BLOCK_BYTES, 0);
      std::vector<uint8_t> h_out(count * DIGEST_BYTES);

      for (size_t i = 0; i < count; i++) {
        memcpy(&h_in[i * BLOCK_BYTES],
             bucket[start + i].data(), len);
        h_in[i * BLOCK_BYTES + len] = 0x80;
        uint64_t bits = len * 8;
        h_in[i * BLOCK_BYTES + 63] = bits & 0xFF;
        h_in[i * BLOCK_BYTES + 62] = (bits >> 8) & 0xFF;
      }

      uint8_t *d_in, *d_out;
      CUDA_CHECK(cudaMalloc(&d_in, h_in.size()));
      CUDA_CHECK(cudaMalloc(&d_out, h_out.size()));
      CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size(),
                  cudaMemcpyHostToDevice));

      int threads = 256;
      int blocks = (count + threads - 1) / threads;

      cudaEvent_t s, e;
      CUDA_CHECK(cudaEventCreate(&s));
      CUDA_CHECK(cudaEventCreate(&e));

      dispatch[len](d_in, d_out, count, blocks, threads);
      CUDA_CHECK(cudaDeviceSynchronize());

      float ms = 0.0f;

      if (bench) {
        CUDA_CHECK(cudaEventRecord(s));

        for (int i = 0; i < BENCH_ITERATIONS; i++)
          dispatch[len](d_in, d_out, count, blocks, threads);

        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

        double avg = ms / BENCH_ITERATIONS;
        double hps = hashes_per_second(count, avg);
        total_hps += hps;

        std::cout << "[GPU " << gpus[gi] << "][len=" << len
              << "] " << static_cast<uint64_t>(hps)
              << " H/s\n";
      }

      cudaFree(d_in);
      cudaFree(d_out);
      cudaEventDestroy(s);
      cudaEventDestroy(e);
    }

    if (bench) {
      std::cout << "[len=" << len << "] TOTAL: "
            << static_cast<uint64_t>(total_hps)
            << " H/s\n";
    }
  }
}

/* end of source */
