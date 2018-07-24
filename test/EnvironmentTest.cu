#include <catch/catch.hpp>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#ifdef RUN_ON_DEVICE

namespace {

struct Saxpy {
  float a;

  Saxpy(float a) : a(a) {}

  __host__ __device__ float operator()(float& x, float& y) { return a * x + y; }
};
}  // namespace


SCENARIO("Can perform SAXPY operation", "[environment][device]") {
  const int N = 10;
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(N);
  thrust::device_vector<float> d_res(N);

  thrust::sequence(d_x.begin(), d_x.end(), 0, 1);
  thrust::fill(d_y.begin(), d_y.end(), 2);

  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_res.begin(), Saxpy(3));

  std::vector<float> result(d_res.begin(), d_res.end());
  CHECK_THAT(result,
             Catch::Matchers::Equals(std::vector<float>{2, 5, 8, 11, 14, 17, 20, 23, 26, 29}));
}

#endif
