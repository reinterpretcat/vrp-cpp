#include <catch/catch.hpp>

#include "config.hpp"
#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "streams/input/SolomonReader.cu"
#include "test_utils/VectorUtils.hpp"

#include <fstream>

using namespace vrp::models;
using namespace vrp::streams;

/// Calculates cartesian distance between two points on plane in 2D.
struct CartesianDistance {
  __host__ __device__
  float operator()(const thrust::tuple<int, int> &left,
                   const thrust::tuple<int, int> &right) {
    auto x = thrust::get<0>(left) - thrust::get<0>(right);
    auto y = thrust::get<1>(left) - thrust::get<1>(right);
    return static_cast<float>(std::sqrt(x * x + y * y));
  }
};

SCENARIO("Can create distances matrix from solomon format.", "[streams]") {
  std::fstream input(SOLOMON_TESTS_PATH "T1.txt");
  Problem problem;

  SolomonReader<CartesianDistance>::read(input, problem);

  CHECK_THAT(vrp::test::copy(problem.distances),
             Catch::Matchers::Equals(std::vector<float>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
}