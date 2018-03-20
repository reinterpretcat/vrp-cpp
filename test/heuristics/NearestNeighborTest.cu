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

SCENARIO("Can complete solution with nearest neighbor method.", "[heuristics][construction]") {
  std::fstream input(SOLOMON_TESTS_PATH "T2.txt");
  Problem problem;
  SolomonReader<CartesianDistance>::read(input, problem);
  Tasks tasks {problem.size()};

  //NearestNeighbor insertion;

}
