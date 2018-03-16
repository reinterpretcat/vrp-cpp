#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "heuristics/CheapestInsertion.cu"
#include "streams/input/SolomonReader.cu"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::heuristics;
using namespace vrp::models;
using namespace vrp::streams;

SCENARIO("Can complete solution with cheapest insertion.", "[heuristics][cheapest]") {
  std::fstream input(SOLOMON_TESTS_PATH "T2.txt");
  Problem problem;
  SolomonReader<CartesianDistance>::read(input, problem);
  Tasks tasks {problem.size()};

  //CheapestInsertion insertion;

}
