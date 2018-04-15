#include <catch/catch.hpp>

#include "config.hpp"

#include "algorithms/Distances.cu"
#include "heuristics/NoTransition.cu"
#include "solver/genetic/Populations.hpp"
#include "streams/input/SolomonReader.hpp"

#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/VectorUtils.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::heuristics;
using namespace vrp::genetic;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

namespace {

struct WithSequentialCustomers {
  std::stringstream operator()() {
    return SolomonBuilder()
        .setTitle("Sequential customers")
        .setVehicle(1, 10)
        .addCustomer({0, 0, 0, 0, 0, 1000, 0})
        .addCustomer({1, 1, 0, 1, 0, 1000, 10})
        .addCustomer({2, 2, 0, 1, 0, 1000, 10})
        .addCustomer({3, 3, 0, 1, 0, 1000, 10})
        .addCustomer({4, 4, 0, 1, 0, 1000, 10})
        .addCustomer({5, 5, 0, 1, 0, 1000, 10})
        .build();
  }
};

template<typename Heuristic>
Tasks createPopulation(std::istream &stream, int populationSize = 3) {
  auto problem = SolomonReader().read(stream, cartesian_distance());
  return create_population<Heuristic>(problem)({populationSize});
}
}

SCENARIO("Can create roots of initial population.",
         "[genetic][population][initial][roots]") {
  auto stream = WithSequentialCustomers()();

  auto population = createPopulation<NoTransition>(stream);

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

SCENARIO("Can create a full initial population.",
         "[genetic][population][initial][solution]") {
  auto stream = WithSequentialCustomers()();

  auto population = createPopulation<NearestNeighbor>(stream);

  CHECK_THAT(vrp::test::copy(population.ids), Catch::Matchers::Equals(std::vector<int>{
      0, 1, 2, 3, 4, 5,
      0, 2, 1, 3, 4, 5,
      0, 3, 2, 1, 4, 5,
  }));
  CHECK_THAT(vrp::test::copy(population.costs), Catch::Matchers::Equals(std::vector<float>{
      0, 1, 2, 3, 4, 5,
      0, 2, 3, 5, 6, 7,
      0, 3, 4, 5, 8, 9,
  }));

  CHECK_THAT(vrp::test::copy(population.times), Catch::Matchers::Equals(std::vector<int>{
      0, 11, 22, 33, 44, 55,
      0, 12, 23, 35, 46, 57,
      0, 13, 24, 35, 48, 59,
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
         "[genetic][population][initial][two_vehicles]") {
  auto stream = SolomonBuilder()
      .setTitle("Exceeded capacity and two vehicles")
      .setVehicle(2, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 3, 0, 1000, 10})
      .addCustomer({2, 2, 0, 3, 0, 1000, 10})
      .addCustomer({3, 3, 0, 3, 0, 1000, 10})
      .addCustomer({4, 4, 0, 2, 0, 1000, 10})
      .addCustomer({5, 5, 0, 2, 0, 1000, 10})
      .build();

  auto population = createPopulation<NearestNeighbor>(stream, 1);

  CHECK_THAT(vrp::test::copy(population.vehicles), Catch::Matchers::Equals(std::vector<int>{
      0, 0, 0, 0, 1, 1,
  }));
  CHECK_THAT(vrp::test::copy(population.capacities), Catch::Matchers::Equals(std::vector<int>{
      10, 7, 4, 1, 8, 6,
  }));
}

SCENARIO("Can use second vehicle within initial population in case of time violation.",
         "[genetic][initial][two_vehicles]") {
  auto stream = SolomonBuilder()
      .setTitle("Exceeded time and two vehicles")
      .setVehicle(2, 10)
      .addCustomer({0, 0,   0, 0, 0, 1000, 0})
      .addCustomer({1, 1,   0, 1, 0, 1000, 10})
      .addCustomer({2, 2,   0, 1, 0, 1000, 10})
      .addCustomer({3, 3,   0, 1, 0, 1000, 10})
      .addCustomer({4, 4,   0, 1, 0, 1000, 10})
      .addCustomer({5, 100, 0, 2, 0, 101,  10})
      .build();

  auto population = createPopulation<NearestNeighbor>(stream, 1);

  CHECK_THAT(vrp::test::copy(population.vehicles), Catch::Matchers::Equals(std::vector<int>{
      0, 0, 0, 0, 0, 1,
  }));
  CHECK_THAT(vrp::test::copy(population.capacities), Catch::Matchers::Equals(std::vector<int>{
      10, 9, 8, 7, 6, 8,
  }));
}
