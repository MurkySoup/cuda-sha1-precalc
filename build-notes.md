## CUDA Architecture (`SM=xx`) Build Options

The `Makefile` allows you to specify a target GPU architecture at build time using the `SM=xx` parameter:

```bash
make SM=86
```

This controls which Streaming Multiprocessor (SM) architecture the CUDA code is compiled for.

Choosing the correct value ensures:

* Optimal performance
* No runtime JIT compilation
* Predictable benchmarking results

---

### What does `SM` mean?

`SM` stands for Streaming Multiprocessor, NVIDIA’s term for the fundamental execution unit on a GPU.

Each GPU generation implements a different SM version, identified by a two-digit number:

```
SM = <major><minor>
```

Example:

* `SM=86` → Compute Capability 8.6 (Ampere, consumer GPUs)

---

### Why this matters

CUDA supports compiling code in several ways:

* Native SASS (best performance, no JIT)
* PTX (portable, JIT-compiled at runtime)

This project intentionally avoids PTX fallback to keep:

* Performance measurements honest
* Behavior reproducible
* Startup overhead minimal

By specifying `SM=xx`, you ensure the binary contains **native code for your GPU**.

---

### Common `SM` values by GPU generation

| GPU Architecture    | Compute Capability | `SM` value | Examples          |
| ------------------- | ------------------ | ---------- | ----------------- |
| Maxwell             | 5.2                | `52`       | GTX 970, 980      |
| Pascal              | 6.1                | `61`       | GTX 1080, Titan X |
| Volta               | 7.0                | `70`       | V100              |
| Turing              | 7.5                | `75`       | RTX 20xx          |
| Ampere (Datacenter) | 8.0                | `80`       | A100              |
| Ampere (Consumer)   | 8.6                | `86`       | RTX 30xx          |
| Ada Lovelace        | 8.9                | `89`       | RTX 40xx          |
| Hopper              | 9.0                | `90`       | H100              |

If unsure, run:

```bash
nvidia-smi
```

or:

```bash
nvcc --list-gpu-arch
```

---

### Choosing the *right* value

Rule of thumb:

* If you are benchmarking → match your exact GPU
* If you are distributing binaries → target the lowest SM you expect
* If you are experimenting → be explicit and intentional

Examples:

```bash
make SM=86   # RTX 3080 / 3090
make SM=89   # RTX 4090
make SM=80   # A100
make SM=90   # H100
```

---

### What happens if you choose incorrectly?

* Higher than your GPU supports: The binary will fail to run.
* Lower than your GPU supports: The code will run, but you may miss:
  * Instruction-level improvements
  * Register file enhancements
  * Scheduler optimizations

This can skew benchmarking results.

---

### Advanced note (intentional design choice)

This project compiles one architecture per build by default.

This is deliberate and creates:
* Smaller binaries
* Faster compile times
* Cleaner performance analysis

If you need multi-architecture ("fat") binaries, the `Makefile` can be extended, but this is not enabled by default to avoid accidental performance ambiguity.

---

### Summary

* `SM=xx` selects the GPU architecture target
* Match it to your hardware for best results
* Being explicit avoids surprises
* Reproducibility beats convenience for research
* *Good documentation is a performance feature for humans.*
