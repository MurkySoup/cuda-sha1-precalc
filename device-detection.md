# CUDA-compatible device detection(s)

When multiple GPUs are present, the helper tools recommend compiling for the highest compute capability common to all devices.

This ensures the resulting binary can execute on every detected GPU without recompilation. Users targeting a specific device may override this manually.

# `detect_sm.sh` — Shell-Based Detector (Zero Compilation)

This version relies on nvidia-smi only, making it ideal for:
* Fresh systems
* Containers
* CI environments
* Users without a CUDA dev toolchain installed yet

This tool includes Mixed Major Architecture Warnings:
* Track SM major (sm / 10)
* Detect if more than one unique major exists
* Emit warning after enumeration, but keeps recommendation unchanged

# `detect_sm.cu` — Minimal Authoritative CUDA Runtime Detector

This version queries the CUDA runtime directly, making it:
* Immune to nvidia-smi quirks
* Accurate in virtualized environments
* Ideal for research-grade reproducibility
* It compiles quickly and has no external dependencies beyond CUDA. `nvcc detect_sm.cu -o detect_sm`

This tool also includes Mixed Major Architecture Warnings:
* Track prop.major
* Detect heterogeneity
* Print warning after recommendation
* Zero performance or runtime cost

# Example output(s)

```yaml
Detected 2 CUDA device(s)

GPU 0: NVIDIA GeForce RTX 3060
  Compute Capability: 8.6
  Individual build: make SM=86

GPU 1: NVIDIA GeForce RTX 3060
  Compute Capability: 8.6
  Individual build: make SM=86

Multiple GPUs detected.
Highest common supported SM: 86
Recommended portable build:
  make SM=86
```

```yaml
Multiple GPUs detected.
Highest common supported SM: 70
Recommended portable build:
  make SM=70

⚠️  Architecture diversity warning:
   Detected GPUs span multiple major SM versions: 7 9
   This may limit performance optimizations and instruction selection.
   Consider per-architecture builds if maximum performance is required.
```

# Why "Highest Common SM" Is the Right Default

This approach:
* Avoids illegal instruction faults
* Allows one binary to run everywhere
* Matches CUDA fat-binary best practices (without fat binaries)
* Keeps build logic simple and explicit
* Scales cleanly to multi-GPU research systems

This is the same trade-off Hashcat, cuBLAS, and NCCL make when portability matters.

# Why "Mixed Major Architecture" Warnings Matters

When GPUs span multiple major compute capability versions (e.g., SM 7.x and SM 9.x), compiling for the lowest common SM can significantly restrict instruction availability, memory model features, and compiler optimizations.

In such cases, per-architecture builds may yield materially different performance characteristics.
