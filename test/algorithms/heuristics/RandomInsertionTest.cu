#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "config.hpp"
#include "streams/input/SolomonReader.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

SCENARIO("Can find transition after depot.", "[heuristics][construction][RandomInsertion][init]") {
  auto stream = create_sequential_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};
  vrp::test::createDepotTask(problem, tasks);

  auto transition = random_insertion(problem.getShadow(), tasks.getShadow(), {})({0, 0, 1, 0});

  // TODO
  // REQUIRE(transition.isValid());
}
