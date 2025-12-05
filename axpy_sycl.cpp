#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

void run_axpy_sycl() {
  std::size_t N = 20'000'000;

  std::vector<float> A(N, 1.0f);
  std::vector<float> B(N, 2.0f);
  std::vector<float> OUT2(N, 0.0f);

  sycl::queue q;
  std::cout << "SYCL device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  auto sycl_start = std::chrono::high_resolution_clock::now();

  {
    sycl::buffer<float> a_buf(A.data(), N);
    sycl::buffer<float> b_buf(B.data(), N);
    sycl::buffer<float> o_buf(OUT2.data(), N);

    q.submit([&](sycl::handler &h) {
       auto a = a_buf.get_access<sycl::access::mode::read>(h);
       auto b = b_buf.get_access<sycl::access::mode::read>(h);
       auto o = o_buf.get_access<sycl::access::mode::write>(h);

       h.parallel_for(N, [=](sycl::id<1> i) { o[i] = a[i] + 2.0f * b[i]; });
     }).wait();
  }

  auto sycl_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sycl_time = sycl_end - sycl_start;

  std::cout << "SYCL time: " << sycl_time.count() << " s\n";
  std::cout << "OUT2[0] = " << OUT2[0] << "\n";
}
