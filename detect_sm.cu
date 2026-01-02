/* nvcc detect_sm.cu -o detect_sm */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <cstdlib>

#define CUDA_CHECK(x) do {                        \
  cudaError_t err = (x);                          \
  if (err != cudaSuccess) {                       \
    std::cerr << cudaGetErrorString(err) << "\n"; \
    std::exit(1);                                 \
  }                                               \
} while (0)

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));

  if (count == 0) {
    std::cerr << "No CUDA-capable devices found\n";
    return 1;
  }

  std::vector<int> sms;
  std::set<int> majors;

  std::cout << "Detected " << count << " CUDA device(s)\n\n";

  for (int i = 0; i < count; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    int sm = prop.major * 10 + prop.minor;
    sms.push_back(sm);
    majors.insert(prop.major);

    std::cout << "GPU " << i << ": " << prop.name << "\n";
    std::cout << "  Compute Capability: "
          << prop.major << "." << prop.minor << "\n";
    std::cout << "  Individual build: make SM=" << sm << "\n\n";
  }

  int common_sm = *std::min_element(sms.begin(), sms.end());

  if (sms.size() > 1) {
    std::cout << "Multiple GPUs detected.\n";
    std::cout << "Highest common supported SM: " << common_sm << "\n";
    std::cout << "Recommended portable build:\n";
    std::cout << "  make SM=" << common_sm << "\n\n";

    if (majors.size() > 1) {
      std::cout << "⚠️  Architecture diversity warning:\n";
      std::cout << "   Detected GPUs span multiple major SM versions: ";
      for (int m : majors) std::cout << m << " ";
      std::cout << "\n";
      std::cout << "   This may limit instruction selection and peak performance.\n";
      std::cout << "   Consider per-architecture builds for benchmarking or deployment.\n";
    }
  } else {
    std::cout << "Single GPU detected.\n";
    std::cout << "Recommended build:\n";
    std::cout << "  make SM=" << sms[0] << "\n";
  }

  return 0;
}

/* end of source */
