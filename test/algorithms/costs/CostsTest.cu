#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/costs/TransitionCosts.hpp"
#include "models/Solution.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::costs;
using namespace vrp::models;
using namespace vrp::test;

namespace {
Solution getPopulation(int populationSize) {
  auto stream = SolomonBuilder()
                  .setTitle("Exceeded capacity and two vehicles")
                  .setVehicle(3, 10)
                  .addCustomer({0, 0, 0, 0, 0, 1000, 0})
                  .addCustomer({1, 1, 0, 8, 0, 1000, 0})
                  .addCustomer({2, 2, 0, 8, 0, 1000, 0})
                  .addCustomer({3, 3, 0, 4, 0, 1000, 0})
                  .addCustomer({4, 4, 0, 3, 0, 1000, 0})
                  .addCustomer({5, 5, 0, 3, 0, 1000, 0})
                  .build();
  return createPopulation<>(stream, populationSize);
};
}  // namespace

SCENARIO("Can calculate total cost for single solution.", "[algorithm][costs]") {
  auto solution = getPopulation(1);

  auto cost = calculate_total_cost()(solution);

  // locations : 0, 1, 2, 3, 4, 5
  // vehicles  : 0, 0, 1, 2, 2, 2
  // costs     : 0, 1, 2, 3, 4, 5
  REQUIRE(cost == 16);
}

SCENARIO("Can calculate total cost for multiple solutions.", "[algorithm][costs]") {
  auto solution = getPopulation(3);

  auto cost = calculate_total_cost()(solution, 1);

  // locations : 0, 3, 4, 5, 1, 2
  // vehicles  : 0, 0, 0, 0, 1, 2
  // costs     : 0, 3, 4, 5, 1, 2
  REQUIRE(cost == 16);
}