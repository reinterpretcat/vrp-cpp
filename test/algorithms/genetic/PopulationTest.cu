#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Dummy.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "config.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

namespace {

const auto T = Plan::assign();
const auto F = Plan::empty();

}  // namespace

SCENARIO("Can create roots of initial population.", "[genetic][population][initial][roots]") {
  auto stream = create_sequential_problem_stream{}();
  auto solution = createPopulation<dummy<TransitionOperator>>(stream);

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
  auto stream = create_sequential_problem_stream{}();
  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream);

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
  auto stream = create_exceeded_capacity_variant_2_problem_stream{}();

  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 1);

  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles),
             Catch::Matchers::Equals(std::vector<int>{0, 0, 0, 0, 1, 1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities),
             Catch::Matchers::Equals(std::vector<int>{10, 7, 4, 1, 8, 6}));
}

SCENARIO("Can use second vehicle within initial population in case of time violation.",
         "[genetic][initial][two_vehicles]") {
  auto stream = create_exceeded_time_problem_stream{}();

  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 1);

  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles),
             Catch::Matchers::Equals(std::vector<int>{0, 0, 0, 0, 0, 1}));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities),
             Catch::Matchers::Equals(std::vector<int>{10, 9, 8, 7, 6, 8}));
}
