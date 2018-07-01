#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

SCENARIO("Can create transition from convolution.", "[transitions][convolutions]") {
  auto stream = create_sequential_problem_stream{}();
  auto solution = createPopulation<nearest_neighbor>(stream, 1);
  thrust::fill(thrust::device, solution.tasks.plan.begin() + 3, solution.tasks.plan.end(),
               Plan::reserve(0));
  vrp::utils::device_variant<int, Convolution> variant;
  variant.set<Convolution>(Convolution{0, 3, 30, {3, 5}, {30, 1000}, {3, 5}});
  auto details = Transition::Details{0, 2, 3, variant, 0};

  auto transition =
    create_transition(solution.problem.getShadow(), solution.tasks.getShadow())(details);

  REQUIRE(transition.isValid());
  REQUIRE(transition.details.customer.is<Convolution>());
  REQUIRE(transition.delta.distance == 1);
  REQUIRE(transition.delta.traveling == 1);
  REQUIRE(transition.delta.serving == 30);
  REQUIRE(transition.delta.waiting == 7);
  REQUIRE(transition.delta.demand == 3);
}
