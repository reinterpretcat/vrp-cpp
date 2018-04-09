#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/Distances.cu"
#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "streams/input/SolomonReader.cu"

#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/VectorUtils.hpp"

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

namespace {
struct WithSimplifiedCoordinates {
  std::stringstream operator()() {
    return SolomonBuilder()
        .setTitle("Customers with simplified coordinates")
        .setVehicle(1, 10)
        .addCustomer({0, 0, 0, 0, 0, 1000, 1})
        .addCustomer({1, 1, 0, 1, 0, 1000, 1})
        .addCustomer({2, 3, 0, 1, 0, 1000, 1})
        .addCustomer({3, 7, 0, 1, 0, 1000, 1})
        .build();
  }
};
}

SCENARIO("Can create customers data.", "[streams][solomon]") {
  auto stream = WithSimplifiedCoordinates()();

  auto problem = SolomonReader<CartesianDistances>::read(stream);

  CHECK_THAT(vrp::test::copy(problem.customers.demands),
             Catch::Matchers::Equals(std::vector<int>{0, 1, 1, 1}));
  CHECK_THAT(vrp::test::copy(problem.customers.services),
            Catch::Matchers::Equals(std::vector<int>(4, 1)));
  CHECK_THAT(vrp::test::copy(problem.customers.starts),
             Catch::Matchers::Equals(std::vector<int>(4, 0)));
  CHECK_THAT(vrp::test::copy(problem.customers.ends),
             Catch::Matchers::Equals(std::vector<int>(4, 1000)));
}

SCENARIO("Can create routing data.", "[streams][solomon]") {
  auto stream = WithSimplifiedCoordinates()();

  auto problem = SolomonReader<CartesianDistances>::read(stream);

  CHECK_THAT(vrp::test::copy(problem.routing.distances),
             Catch::Matchers::Equals(std::vector<float>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
  CHECK_THAT(vrp::test::copy(problem.routing.durations),
             Catch::Matchers::Equals(std::vector<int>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
}

SCENARIO("Can create resources data.", "[streams][solomon]") {
  auto stream = WithSimplifiedCoordinates()();

  auto problem = SolomonReader<CartesianDistances>::read(stream);

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