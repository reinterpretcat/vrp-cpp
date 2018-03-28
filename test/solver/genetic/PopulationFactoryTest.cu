#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "solver/genetic/PopulationFactory.hpp"
#include "streams/input/SolomonReader.cu"
#include "streams/output/ContainerWriter.hpp"

#include "test_utils/VectorUtils.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::genetic;
using namespace vrp::models;
using namespace vrp::streams;

SCENARIO("Can create initial population.", "[genetic][initial][T2]") {
  std::fstream input(SOLOMON_TESTS_PATH "T2.txt");
  Settings settings{3};
  auto problem = SolomonReader<CartesianDistance>::read(input);

  auto population = createPopulation(problem, settings);


  CHECK_THAT(vrp::test::copy(population.ids), Catch::Matchers::Equals(std::vector<int>{
      0, 1, -1, -1, -1, -1,
      0, 2, -1, -1, -1, -1,
      0, 3, -1, -1, -1, -1,
  }));
  CHECK_THAT(vrp::test::copy(population.costs), Catch::Matchers::Equals(std::vector<float>{
      0, 1, -1, -1, -1, -1,
      0, 2, -1, -1, -1, -1,
      0, 3, -1, -1, -1, -1,
  }));

  CHECK_THAT(vrp::test::copy(population.times), Catch::Matchers::Equals(std::vector<int>{
      0, 11, -1, -1, -1, -1,
      0, 12, -1, -1, -1, -1,
      0, 13, -1, -1, -1, -1,
  }));
  CHECK_THAT(vrp::test::copy(population.capacities), Catch::Matchers::Equals(std::vector<int>{
      10, 9, -1, -1, -1, -1,
      10, 9, -1, -1, -1, -1,
      10, 9, -1, -1, -1, -1,
  }));
  CHECK_THAT(vrp::test::copy(population.vehicles), Catch::Matchers::Equals(std::vector<int>{
      0, 0, -1, -1, -1, -1,
      0, 0, -1, -1, -1, -1,
      0, 0, -1, -1, -1, -1,
  }));
  CHECK_THAT(vrp::test::copy(population.plan), Catch::Matchers::Equals(std::vector<bool>{
      true, true, false, false, false, false,
      true, false, true, false, false, false,
      true, false, false, true, false, false,
  }));
}
