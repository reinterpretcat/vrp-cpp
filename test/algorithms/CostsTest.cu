#include <catch/catch.hpp>

#include "algorithms/Costs.cu"
#include "models/Tasks.hpp"

#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/VectorUtils.hpp"

using namespace vrp::algorithms;
using namespace vrp::test;

SCENARIO("Can calculate vehicles cost from solution.", "[algorithm][costs]") {
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

  auto costs = calculate_vehicles_cost()(createPopulation<>(stream, 1));

  // vehicles: 0, 0, 1, 2, 2, 2
  // costs   : 0, 1, 2, 3, 4, 5
  CHECK_THAT(vrp::test::copy(costs), Catch::Matchers::Equals(std::vector<float>{
      1, 2, 5
  }));
}