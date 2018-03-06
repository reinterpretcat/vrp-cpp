#include <catch/catch.hpp>

#include "streams/output/VectorWriter.hpp"
#include "test_utils/VectorUtils.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace {

struct Saxpy {
  float a;

  Saxpy(float a) : a(a) {}

  __host__ __device__
  float operator()(float &x, float &y) {
    return a * x + y;
  }
};
}

SCENARIO("Can perform SAXPY operation", "[environment]") {
  const int N = 10;
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(N);
  thrust::device_vector<float> d_res(N);

  thrust::sequence(d_x.begin(), d_x.end(), 0, 1);
  thrust::fill(d_y.begin(), d_y.end(), 2);

  thrust::transform(d_x.begin(), d_x.end(),
                    d_y.begin(),
                    d_res.begin(),
                    Saxpy(3));

  CHECK_THAT(vrp::test::copy(d_res),
             Catch::Matchers::Equals(std::vector<float>{ 2, 5, 8, 11, 14, 17, 20, 23, 26, 29 }));
}
