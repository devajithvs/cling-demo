#include <sycl/sycl.hpp>
#include <iostream>

void sycl_example(sycl::queue &q, size_t N) {
    int* data = sycl::malloc_shared<int>(N, q);
    int* out  = sycl::malloc_shared<int>(N, q);

    // Initialize input
    for (size_t i = 0; i < N; i++)
        data[i] = 1;

    // Kernel: out[i] = data[i] * 2
    q.parallel_for(N, [=](sycl::id<1> i) {
        out[i] = data[i] * 2;
    }).wait();

    std::cout << "Result: out[0] = " << out[0] << "\n";

    sycl::free(data, q);
    sycl::free(out, q);
}

