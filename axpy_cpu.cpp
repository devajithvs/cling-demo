#include <chrono>
#include <iostream>
#include <vector>

void run_axpy_cpu() {
  std::size_t N = 20'000'000;

  std::vector<float> A(N, 1.0f);
  std::vector<float> B(N, 2.0f);
  std::vector<float> OUT(N, 0.0f);

  auto cpu_start = std::chrono::high_resolution_clock::now();

  for (std::size_t i = 0; i < N; ++i)
    OUT[i] = A[i] + 2.0f * B[i];

  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpu_time = cpu_end - cpu_start;

  std::cout << "CPU time: " << cpu_time.count() << " s\n";
  std::cout << "OUT[0] = " << OUT[0] << "\n";
}
