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
* Selectability of specific GPUs or use all available devices
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

## Design Overview and Constraints

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

This project is also intentionally constrained. Those constraints are not accidents — they are design decisions. The codebase is designed to:
* Favor clarity over maximal cleverness
* Be auditable by a single experienced developer
* Use explicit kernel specialization rather than runtime polymorphism
* Prefer compile-time decisions over complex runtime logic
* Avoid framework-level abstractions that obscure GPU behavior
* Remain usable as a research and learning artifact, not just a benchmark

Performance optimizations are chosen only when they:
* Are well-understood
* Have clear justification
* Do not significantly increase maintenance burden

---

# Non-Goals and What This Project *Is Not*

* A password cracker
* A drop-in replacement for Hashcat
* A production cryptographic library
* An attempt to "win" SHA-1 benchmarks at all costs

Those problems are already well-solved elsewhere.

This project *does not* attempt to:
* Replace or compete directly with production tools like Hashcat
* Implement every known GPU micro-optimization
* Automatically tune kernels per device at runtime
* Provide a full password-cracking framework
* Hide hardware realities behind abstraction layers
* Optimize for legacy GPUs beyond basic compatibility

If your goal is:
* Maximum throughput at any cost → use a production cracker (Hashcat, JohntheRipper, etc.)
* Broad algorithm coverage → use an existing framework
* Academic exploration and controlled optimization → you’re in the right place

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

See `device-detection.md` and `build-notes.md` for more more details of available build options and device detection tools.

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

# Benchmark Methodology Notes

Benchmarking GPU code is deceptively easy to do incorrectly. This project includes a simple benchmarking mode intended for comparative research and sanity checking, not marketing-grade performance claims.

## What the benchmark measures

The benchmark primarily reports:
* Hashes per second
* End-to-end kernel execution time
* Host ↔ device transfer overhead (when enabled)
* Benchmarks are intended to compare:
* Kernel variants within this codebase
* Architectural effects (SM versions, occupancy changes)
* Design trade-offs (portability vs specialization)

## What the benchmark explicitly does not claim

Results from this codebase should not be directly compared to:
* Hashcat “best case” numbers
* Vendor marketing benchmarks
* Results using algorithm-specific assembly
* Highly hand-tuned, architecture-exclusive kernels
* Benchmarks using multiple overlapping optimizations simultaneously

Those tools optimize under very different constraints and conditions.

## Common benchmarking pitfalls (and how we avoid them)

| Rookie Mistake                                                   | Pro Design Tip                                   |
|------------------------------------------------------------------|--------------------------------------------------|
| Comparing different input distributions                          | Inputs are fixed and repeatable                  |
| Measuring compilation-time specialization as runtime performance | Kernel specialization is explicit and documented |
| Ignoring GPU warm-up effects                                     | Benchmarks run multiple iterations               |
| Overlooking mixed-architecture systems                           | SM detection warns about heterogeneous GPUs      |

## Recommended benchmarking practices

For meaningful results:
* Pin execution to a specific GPU
* Ensure no other GPU workloads are active
* Use the same input set across runs
* Report SM version, GPU model, driver, and CUDA version
* State whether results are portable or architecture-specific

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

SHA-1 is cryptographically broken since the early 2000's and should not be used for security-critical applications. See: https://en.wikipedia.org/wiki/SHA-1

---

## Closing Thoughts

This project is meant to be interesting to read, safe to modify, and fast enough to matter.

If it helps you learn something, test an idea, or ask better questions — then it’s doing its job.

Comments, suggestions, and constructive critique are welcome.

# License

This tool is released under the Apache 2.0 license. See the LICENSE file in this repo for details.

# Built With

* [nVidia CUDA](https://developer.nvidia.com/cuda-downloads)

## Author

**Rick Pelletier** - [Gannett Co., Inc. (USA Today Network)](https://www.usatoday.com/)
