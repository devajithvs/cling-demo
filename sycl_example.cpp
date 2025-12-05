#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

void run_sycl_demo() {
  sycl::queue q;
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  const size_t N = 1024;
  std::vector<int> data(N, 1);
  std::vector<int> out(N);

  {
    sycl::buffer<int> b_in(data.data(), N);
    sycl::buffer<int> b_out(out.data(), N);

    q.submit([&](sycl::handler &h) {
       auto in = b_in.get_access<sycl::access::mode::read>(h);
       auto out = b_out.get_access<sycl::access::mode::write>(h);

       h.parallel_for(N, [=](sycl::id<1> i) { out[i] = in[i] * 2; });
     }).wait();
  }

  std::cout << "out[0] = " << out[0] << "\n";
}
