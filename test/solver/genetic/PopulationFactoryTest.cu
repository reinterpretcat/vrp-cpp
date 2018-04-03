#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/CartesianDistance.cu"
#include "heuristics/NoTransition.cu"
#include "solver/genetic/Populations.hpp"
#include "streams/input/SolomonReader.cu"
#include "streams/output/ContainerWriter.hpp"

#include "test_utils/VectorUtils.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::heuristics;
using namespace vrp::genetic;
using namespace vrp::models;
using namespace vrp::streams;

template <typename Heuristic>
Tasks createPopulation(const char *problemFile, int populationSize = 3) {
  std::fstream input(problemFile);
  Settings settings{populationSize};
  auto problem = SolomonReader<CartesianDistance>::read(input);
  return create_population<Heuristic>(problem)(settings);
}

SCENARIO("Can create roots of initial population.", "[genetic][initial][roots][T2]") {
  auto population = createPopulation<NoTransition>(SOLOMON_TESTS_PATH "T2.txt");

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

SCENARIO("Can create a full initial population.", "[genetic][initial][solution][T2]") {
  auto population = createPopulation<NearestNeighbor>(SOLOMON_TESTS_PATH "T2.txt");

  CHECK_THAT(vrp::test::copy(population.ids), Catch::Matchers::Equals(std::vector<int>{
      0, 1, 2, 3, 4, 5,
      0, 2, 1, 3, 4, 5,
      0, 3, 2, 1, 4, 5,
  }));
  CHECK_THAT(vrp::test::copy(population.costs), Catch::Matchers::Equals(std::vector<float>{
      0, 1, 1, 1, 1, 1,
      0, 2, 1, 2, 1, 1,
      0, 3, 1, 1, 3, 1,
  }));

  CHECK_THAT(vrp::test::copy(population.times), Catch::Matchers::Equals(std::vector<int>{
      0, 11, 11, 11, 11, 11,
      0, 12, 11, 12, 11, 11,
      0, 13, 11, 11, 13, 11,
  }));
  CHECK_THAT(vrp::test::copy(population.capacities), Catch::Matchers::Equals(std::vector<int>{
      10, 9, 8, 7, 6, 5,
      10, 9, 8, 7, 6, 5,
      10, 9, 8, 7, 6, 5,
  }));
  CHECK_THAT(vrp::test::copy(population.vehicles), Catch::Matchers::Equals(std::vector<int>{
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
  }));
  CHECK_THAT(vrp::test::copy(population.plan), Catch::Matchers::Equals(std::vector<bool>{
      true, true, true, true, true, true,
      true, true, true, true, true, true,
      true, true, true, true, true, true,
  }));
}

SCENARIO("Can use second vehicle within initial population in case of demand violation.",
         "[genetic][initial][two_vehicles][capacity_check][T4]") {
  auto population = createPopulation<NearestNeighbor>(SOLOMON_TESTS_PATH "T4.txt", 1);

  CHECK_THAT(vrp::test::copy(population.vehicles), Catch::Matchers::Equals(std::vector<int>{
      0, 0, 0, 0, 1, 1,
  }));
  CHECK_THAT(vrp::test::copy(population.capacities), Catch::Matchers::Equals(std::vector<int>{
      10, 7, 4, 1, 8, 6,
  }));
}

SCENARIO("Can use second vehicle within initial population in case of time violation.",
         "[genetic][initial][two_vehicles][time_check][T4]") {
  auto population = createPopulation<NearestNeighbor>(SOLOMON_TESTS_PATH "T5.txt", 1);

  CHECK_THAT(vrp::test::copy(population.vehicles), Catch::Matchers::Equals(std::vector<int>{
      0, 0, 0, 0, 0, 1,
  }));
  CHECK_THAT(vrp::test::copy(population.capacities), Catch::Matchers::Equals(std::vector<int>{
      10, 9, 8, 7, 6, 8,
  }));
}
