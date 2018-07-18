#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::runtime;
using namespace vrp::test;

namespace {
const auto T = Plan::assign();

}  // namespace

SCENARIO("Can execute transition with convolution.", "[transitions][executors][convolutions]") {
  int capacity = 10;
  auto stream = create_sequential_problem_stream{}(capacity);
  auto solution = createPopulation<nearest_neighbor>(stream, 1);
  auto expectedIds = vrp::test::copy(solution.tasks.ids);
  auto expectedVehicles = vrp::test::copy(solution.tasks.vehicles);
  auto expectedCosts = vrp::test::copy(solution.tasks.costs);
  auto expectedCap = vrp::test::copy(solution.tasks.capacities);
  auto expectedTimes = vrp::test::copy(solution.tasks.times);
  auto expectedPlan = vrp::test::copy(solution.tasks.plan);
  thrust::fill(thrust::device, solution.tasks.plan.begin() + 3, solution.tasks.plan.end(),
               Plan::reserve(0));
  auto convolution = Convolution{0, 3, 30, {3, 5}, {30, 1000}, {3, 5}};
  device_variant<int, Convolution> variant;
  variant.set<Convolution>(convolution);
  auto transition = Transition{Transition::Details{0, 2, 3, variant, 0}, {}};

  perform_transition(solution.problem.getShadow(), solution.tasks.getShadow())(transition, 0);

  CHECK_THAT(vrp::test::copy(solution.tasks.ids), Catch::Matchers::Equals(expectedIds));
  CHECK_THAT(vrp::test::copy(solution.tasks.vehicles), Catch::Matchers::Equals(expectedVehicles));
  CHECK_THAT(vrp::test::copy(solution.tasks.costs), Catch::Matchers::Equals(expectedCosts));
  CHECK_THAT(vrp::test::copy(solution.tasks.capacities), Catch::Matchers::Equals(expectedCap));
  CHECK_THAT(vrp::test::copy(solution.tasks.times), Catch::Matchers::Equals(expectedTimes));
  CHECK_THAT(vrp::test::copy(solution.tasks.plan), Catch::Matchers::Equals(expectedPlan));
}
