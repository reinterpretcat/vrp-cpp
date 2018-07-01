#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::utils;
using namespace vrp::test;

namespace {

struct create_problem_stream {
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

}  // namespace

SCENARIO("Can check valid solution", "[utils][validation][solution_checker]") {
  auto stream = create_problem_stream()();
  auto population = createPopulation<nearest_neighbor>(stream, 2);

  auto result = SolutionChecker::check(population);

  REQUIRE(result.isValid());
}