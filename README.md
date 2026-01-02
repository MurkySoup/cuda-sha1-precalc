# CUDA SHA-1 Length-Bucketed Hashing (Research POC)

This repository contains a research-oriented CUDA-based proof-of-concept for computing SHA-1 hashes on NVIDIA GPUs, optimized around warp-level efficiency, fixed-length kernels, and maintainable design.

The project explores how far careful architectural choices (length bucketing, compile-time specialization, and clean dispatch) can go without turning the codebase into an unmaintainable science experiment.

It is not intended to replace mature tools like Hashcat — rather, it exists to support learning, experimentation, benchmarking, and future research.

---

## Key Features

* CUDA 11+ / nvcc
* Length bucketing (0–55 bytes)
* Compile-time fixed-length SHA-1 kernels
* Auto-generated kernel dispatch table
* Host-side padding & normalization
* Warp-friendly execution (no divergence)
* Built-in benchmarking mode (hashes/sec)
* Multi-GPU support

* Select specific GPUs or use all available devices
* Minimal dependencies
* Readable, auditable source code

---

## Why This Exists

GPU hashing code often falls into one of two categories:

* Trivial examples that don’t reflect real performance concerns
* Highly optimized frameworks that are difficult to study or modify

This project sits deliberately in the middle:

* Performance-aware
* Explicit in its design choices
* Easy to reason about
* Suitable as a base for further research (e.g., SHA-256, alternative scheduling, warp-cooperative designs)

---

## Building

### Requirements

* NVIDIA GPU
* CUDA Toolkit 11 or newer
* `nvcc`
* A C++17-capable host compiler

### Build

```bash
make
```

Or specify an architecture explicitly:

```bash
make SM=86   # Ampere consumer GPUs
make SM=90   # Hopper
```

---

## Usage

### Basic hashing

```bash
./sha1_bucketed_dispatch_bench input.txt
```

Each line in `input.txt` is treated as a separate message (up to 55 bytes).

Output format:

```
<sha1_hash>  <input>
```

---

### Benchmark mode

```bash
./sha1_bucketed_dispatch_bench input.txt --bench
```

Example output:

```
[GPU 0][len=6]  430012345 H/s
[GPU 1][len=6]  428901234 H/s
[len=6] TOTAL: 858913579 H/s
```

Timing uses CUDA events and measures kernel execution only
(no file I/O, no memory allocation, no printing).

---

### Multi-GPU selection

Use a specific GPU:

```bash
./sha1_bucketed_dispatch_bench input.txt --bench --gpu 0
```

Use multiple GPUs:

```bash
./sha1_bucketed_dispatch_bench input.txt --bench --gpu 0,1
```

Use all available GPUs:

```bash
./sha1_bucketed_dispatch_bench input.txt --bench --gpu all
```

If `--gpu` is omitted, GPU 0 is used by default.

---

## Design Overview (High-Level)

* Inputs are bucketed by exact length on the host
* Each length maps to a compile-time specialized kernel
* A constexpr dispatch table selects the kernel
* Each GPU operates independently on its assigned slice
* Results and benchmark metrics are aggregated on the host

There is:

* No inter-GPU communication
* No unified memory
* No device-side synchronization between GPUs

This keeps the behavior deterministic and debuggable.

---

## What This Is *Not*

* A password cracker
* A drop-in replacement for Hashcat
* A production cryptographic library
* An attempt to “win” SHA-1 benchmarks at all costs

Those problems are already well-solved elsewhere.

---

## Research Directions (Future Work)

Some natural extensions that fit this codebase well:

* SHA-256 / SHA-512 kernels
* Warp-cooperative message scheduling
* Dynamic load balancing across heterogeneous GPUs
* Power-efficiency measurements (H/s/W)
* NUMA-aware CPU↔GPU affinity
* Multi-stream overlap of transfer and compute

These are intentionally not implemented yet.

---

## Security Note

This code is provided for research and educational purposes only.

SHA-1 is cryptographically broken and should not be used for security-critical applications.

---

## Closing Thoughts

This project is meant to be interesting to read, safe to modify, and fast enough to matter.

If it helps you learn something, test an idea, or ask better questions — then it’s doing its job.

Comments, suggestions, and constructive critique are welcome.


# License

This tool is released under the MIT license. See the LICENSE file in this repo for details.

# Built With

* [nVidia CUDA](https://developer.nvidia.com/cuda-downloads)

## Author

**Rick Pelletier** - [Gannett Co., Inc. (USA Today Network)](https://www.usatoday.com/)
