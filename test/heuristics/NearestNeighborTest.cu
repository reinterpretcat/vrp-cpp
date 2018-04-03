#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "heuristics/NearestNeighbor.hpp"
#include "streams/input/SolomonReader.cu"

#include "test_utils/TaskUtils.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::heuristics;
using namespace vrp::models;
using namespace vrp::streams;

SCENARIO("Can find best transition after depot.", "[heuristics][construction][nearest_neighbor][T3]") {
  std::fstream input(SOLOMON_TESTS_PATH "T3.txt");
  auto problem = SolomonReader<CartesianDistance>::read(input);
  Tasks tasks {problem.size()};
  vrp::test::createDepotTask(problem, tasks);

  auto transitionCost = NearestNeighbor (problem.getShadow(), tasks.getShadow())(0, 0);

  REQUIRE(thrust::get<0>(transitionCost).details.customer == 3);
  REQUIRE(thrust::get<1>(transitionCost) == 1);
}
