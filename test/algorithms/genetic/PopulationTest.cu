#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Dummy.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "config.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::genetic;
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

const auto T = Plan::assign();
const auto F = Plan::empty();

}  // namespace

SCENARIO("Can create roots of initial population.", "[genetic][population][initial][roots]") {
  auto stream = WithSequentialCustomers()();
  auto solution = createPopulation<dummy>(stream);

  CHECK_THAT(vrp::test::copy(solution.tasks.ids),
             Catch::Matchers::Equals(
               std::vector<int>{0, 1, -1, -1, -1, -1, 0, 3, -1, -1, -1, -1, 0, 5, -1, -1, -1, -1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.costs),
             Catch::Matchers::Equals(std::vector<float>{0, 1, -1, -1, -1, -1, 0, 3, -1, -1, -1, -1,
                                                        0, 5, -1, -1, -1, -1}));

  CHECK_THAT(vrp::test::copy(solution.tasks.times),
             Catch::Matchers::Equals(std::vector<int>{0, 11, -1, -1, -1, -1, 0, 13, -1, -1, -1, -1,
                                                      0, 15, -1, -1, -1, -1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities),
             Catch::Matchers::Equals(std::vector<int>{10, 9, -1, -1, -1, -1, 10, 9, -1, -1, -1, -1,
                                                      10, 9, -1, -1, -1, -1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles),
             Catch::Matchers::Equals(
               std::vector<int>{0, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.plan),
             Catch::Matchers::Equals(
               std::vector<Plan>{T, T, F, F, F, F, T, F, F, T, F, F, T, F, F, F, F, T}));
}

SCENARIO("Can create a full initial population.", "[genetic][population][initial][solution]") {
  auto stream = WithSequentialCustomers()();
  auto solution = createPopulation<nearest_neighbor>(stream);

  CHECK_THAT(vrp::test::copy(solution.tasks.ids),
             Catch::Matchers::Equals(
               std::vector<int>{0, 1, 2, 3, 4, 5, 0, 3, 2, 1, 4, 5, 0, 5, 4, 3, 2, 1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.costs),
             Catch::Matchers::Equals(
               std::vector<float>{0, 1, 2, 3, 4, 5, 0, 3, 4, 5, 8, 9, 0, 5, 6, 7, 8, 9}));

  CHECK_THAT(vrp::test::copy(solution.tasks.times),
             Catch::Matchers::Equals(std::vector<int>{0, 11, 22, 33, 44, 55, 0, 13, 24, 35, 48, 59,
                                                      0, 15, 26, 37, 48, 59}));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities),
             Catch::Matchers::Equals(
               std::vector<int>{10, 9, 8, 7, 6, 5, 10, 9, 8, 7, 6, 5, 10, 9, 8, 7, 6, 5}));
  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles),
             Catch::Matchers::Equals(
               std::vector<int>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  CHECK_THAT(vrp::test::copy(solution.tasks.plan),
             Catch::Matchers::Equals(
               std::vector<Plan>{T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T}));
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

  auto solution = createPopulation<nearest_neighbor>(stream, 1);

  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles),
             Catch::Matchers::Equals(std::vector<int>{0, 0, 0, 0, 1, 1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities),
             Catch::Matchers::Equals(std::vector<int>{10, 7, 4, 1, 8, 6}));
}

SCENARIO("Can use second vehicle within initial population in case of time violation.",
         "[genetic][initial][two_vehicles]") {
  auto stream = SolomonBuilder()
                  .setTitle("Exceeded time and two vehicles")
                  .setVehicle(2, 10)
                  .addCustomer({0, 0, 0, 0, 0, 1000, 0})
                  .addCustomer({1, 1, 0, 1, 0, 1000, 10})
                  .addCustomer({2, 2, 0, 1, 0, 1000, 10})
                  .addCustomer({3, 3, 0, 1, 0, 1000, 10})
                  .addCustomer({4, 4, 0, 1, 0, 1000, 10})
                  .addCustomer({5, 100, 0, 2, 0, 101, 10})
                  .build();

  auto solution = createPopulation<nearest_neighbor>(stream, 1);

  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles),
             Catch::Matchers::Equals(std::vector<int>{0, 0, 0, 0, 0, 1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities),
             Catch::Matchers::Equals(std::vector<int>{10, 9, 8, 7, 6, 8}));
}
