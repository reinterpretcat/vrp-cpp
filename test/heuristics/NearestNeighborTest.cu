#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "config.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/TaskUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

namespace {
struct WithShuffledCoordinates {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Customers with shuffled coordinates")
      .setVehicle(1, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 2, 0, 1, 0, 1000, 10})
      .addCustomer({2, 4, 0, 1, 0, 1000, 10})
      .addCustomer({3, 1, 0, 1, 0, 1000, 10})
      .addCustomer({4, 5, 0, 1, 0, 1000, 10})
      .addCustomer({5, 3, 0, 1, 0, 1000, 10})
      .build();
  }
};
}  // namespace

SCENARIO("Can find best transition after depot.", "[heuristics][construction][NearestNeighbor]") {
  auto stream = WithShuffledCoordinates()();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};
  vrp::test::createDepotTask(problem, tasks);

  auto transitionCost = nearest_neighbor(problem.getShadow(), tasks.getShadow())(0, 0, 0);

  REQUIRE(thrust::get<0>(transitionCost).details.customer == 3);
  REQUIRE(thrust::get<1>(transitionCost) == 1);
}
