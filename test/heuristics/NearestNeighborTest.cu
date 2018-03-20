#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "heuristics/NearestNeighbor.hpp"
#include "streams/input/SolomonReader.cu"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::heuristics;
using namespace vrp::models;
using namespace vrp::streams;

SCENARIO("Can find best transition after depot.", "[heuristics][construction][T3]") {
  std::fstream input(SOLOMON_TESTS_PATH "T3.txt");
  auto problem = SolomonReader<CartesianDistance>::read(input);
  Tasks tasks {problem.size()};
  //NearestNeighbor insertion {problem.getShadow(), tasks.getShadow()};

}
