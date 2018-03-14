#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "solver/genetic/PopulationFactory.hpp"
#include "streams/input/SolomonReader.cu"

#include "test_utils/VectorUtils.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::genetic;
using namespace vrp::models;
using namespace vrp::streams;

SCENARIO("Can create roots.", "[genetic][T2]") {
  std::fstream input(SOLOMON_TESTS_PATH "T2.txt");
  Problem problem;
  SolomonReader<CartesianDistance>::read(input, problem);
  Settings settings { problem.size() };

  auto population = createPopulation(problem, settings);

  //CHECK_THAT(vrp::test::copy(population.ids),
  //           Catch::Matchers::Equals(std::vector<int>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
}

/*SCENARIO("Can create roots with operating time violation.", "[genetic][T2]") {

}

SCENARIO("Can create roots with demand violation.", "[genetic][T2]") {

}*/