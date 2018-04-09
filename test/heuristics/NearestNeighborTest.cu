#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/Distances.cu"
#include "heuristics/NearestNeighbor.hpp"
#include "streams/input/SolomonReader.cu"

#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/TaskUtils.hpp"

using namespace vrp::algorithms;
using namespace vrp::heuristics;
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
}

SCENARIO("Can find best transition after depot.", "[heuristics][construction][nearest_neighbor]") {
  auto stream =  WithShuffledCoordinates()();
  auto problem = SolomonReader<CartesianDistances>::read(stream);
  Tasks tasks {problem.size()};
  vrp::test::createDepotTask(problem, tasks);

  auto transitionCost = NearestNeighbor (problem.getShadow(), tasks.getShadow())(0, 0, 0);

  REQUIRE(thrust::get<0>(transitionCost).details.customer == 3);
  REQUIRE(thrust::get<1>(transitionCost) == 1);
}
