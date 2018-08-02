#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "config.hpp"
#include "streams/input/SolomonReader.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

SCENARIO("Can find best transition after depot.",
         "[heuristics][construction][NearestNeighbor][init]") {
  auto stream = create_shuffled_coordinates{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};
  vrp::test::createDepotTask(problem, tasks);
  auto context = Context{problem.getShadow(), tasks.getShadow(), {}};

  auto transition = nearest_neighbor<TransitionOperator>{}(context, 0, 0, 1, 0);

  REQUIRE(transition.details.customer.get<int>() == 3);
}

SCENARIO("Can find best transition on solution using convolutions.",
         "[heuristics][construction][NearestNeighbor][convolutions]") {
  auto stream = create_shuffled_coordinates{}();
  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 1);
  thrust::fill(exec_unit, solution.tasks.plan.begin() + 3, solution.tasks.plan.end(),
               Plan::reserve(0));
  auto convolution = Convolution{0, 3, 30, {5, 4}, {0, 35}, {3, 5}};
  auto convolutions = create({convolution});
  auto context =
    Context{solution.problem.getShadow(), solution.tasks.getShadow(), convolutions.data()};

  auto transition = nearest_neighbor<TransitionOperator>{}(context, 0, 0, 1, 0);

  // TODO consider cost without service time
  REQUIRE(transition.isValid());
  REQUIRE(transition.details.customer.is<Convolution>());
  compare(transition.details.customer.get<Convolution>(), convolution);
}
