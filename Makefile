# =========================
# Build configuration
# =========================

# Build for default architecture (SM=80)
#   make
#
# Build for Ampere consumer GPUs (e.g., RTX 30xx)
#   make SM=86
#
# Build for Hopper (H100)
#   make SM=90
#
# Clean-up
#   make clean

TARGET      := sha1_bucketed_dispatch_mgpu_bench
SRC         := sha1_bucketed_dispatch_mgpu_bench.cu

NVCC        := nvcc
CXXFLAGS    := -O3 -std=c++17
NVCCFLAGS   := $(CXXFLAGS)

# Set a reasonable default architecture.

# Override on the command line if needed:
#   make SM=86
SM          ?= 80
ARCH        := -gencode arch=compute_$(SM),code=sm_$(SM)

# Optional tuning / diagnostics
# Uncomment as needed
# NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -lineinfo
NVCCFLAGS += -maxrregcount=64

# =========================
# Targets
# =========================

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean

# end of Makefile
