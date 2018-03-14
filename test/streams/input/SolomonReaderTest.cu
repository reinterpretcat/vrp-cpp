#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "streams/input/SolomonReader.cu"

#include "test_utils/VectorUtils.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::streams;


SCENARIO("Can create routing data.", "[streams][solomon][T1]") {
  std::fstream input(SOLOMON_TESTS_PATH "T1.txt");
  Problem problem;

  SolomonReader<CartesianDistance>::read(input, problem);

  CHECK_THAT(vrp::test::copy(problem.routing.distances),
             Catch::Matchers::Equals(std::vector<float>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
  CHECK_THAT(vrp::test::copy(problem.routing.durations),
             Catch::Matchers::Equals(std::vector<int>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
}

SCENARIO("Can create resources data.", "[streams][solomon][T1]") {
  std::fstream input(SOLOMON_TESTS_PATH "T1.txt");
  Problem problem;

  SolomonReader<CartesianDistance>::read(input, problem);

  CHECK_THAT(vrp::test::copy(problem.resources.capacities),
             Catch::Matchers::Equals(std::vector<int>{ 10 }));
  CHECK_THAT(vrp::test::copy(problem.resources.distanceCosts),
             Catch::Matchers::Equals(std::vector<float>{ 1 }));
  CHECK_THAT(vrp::test::copy(problem.resources.timeCosts),
             Catch::Matchers::Equals(std::vector<float>{ 0 }));
  CHECK_THAT(vrp::test::copy(problem.resources.waitingCosts),
             Catch::Matchers::Equals(std::vector<float>{ 0 }));
  CHECK_THAT(vrp::test::copy(problem.resources.timeLimits),
             Catch::Matchers::Equals(std::vector<int>{ std::numeric_limits<int>::max() }));
}