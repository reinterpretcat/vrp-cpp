#include "config.hpp"
#include <catch/catch.hpp>

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "input/SolomonReader.cu"

#include <fstream>

using namespace vrp::models;
using namespace vrp::input;

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

SCENARIO("Can create distances matrix from solomon format.", "[input]") {
  std::fstream input(SOLOMON_DATA_PATH "T1.txt");
  Problem problem;

  SolomonReader<CartesianDistance>::read(input, problem);

  std::vector<float> distances;
  thrust::copy(problem.distances.begin(), problem.distances.end(), std::back_inserter(distances));
  CHECK_THAT(distances, Catch::Matchers::Equals(
      std::vector<float>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
}