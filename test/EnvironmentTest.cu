#include "utils/StreamUtils.hpp"

#include <catch/catch.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

using namespace vrp::utils;

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
  thrust::fill(d_y.begin(), d_y.end(), 0.5);

  thrust::transform(d_x.begin(), d_x.end(),
                    d_y.begin(),
                    d_res.begin(),
                    Saxpy(1.2));

  thrust::host_vector<float> h_res = d_res;
}
