#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "test_utils/TaskUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

namespace {
struct create_shuffled_coordinates {
  std::stringstream operator()(int capacity) {
    return SolomonBuilder()
      .setTitle("Customers with sequential coordinates")
      .setVehicle(1, capacity)
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

SCENARIO("Can create transition from convolution.", "[transitions][convolutions]") {
  int capacity = 10;
  auto stream = create_shuffled_coordinates()(capacity);
  auto solution = createPopulation<nearest_neighbor>(stream, 1);
  thrust::fill(thrust::device, solution.tasks.plan.begin() + 3, solution.tasks.plan.end(),
               Plan::reserve(0));
  vrp::utils::device_variant<int, Convolution> variant;
  variant.set<Convolution>(Convolution{0, 3, 30, {3, 5}, {30, 1000}, {3, 5}});
  auto details = Transition::Details{2, 3, variant, 0};

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
