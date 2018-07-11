#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "config.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <algorithms/transitions/Executors.hpp>
#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::utils;
using namespace vrp::test;

namespace {
typedef typename vrp::algorithms::heuristics::TransitionDelegate<
  vrp::algorithms::transitions::create_transition,
  vrp::algorithms::costs::calculate_transition_cost,
  vrp::algorithms::transitions::perform_transition>
  Delegate;
}

SCENARIO("Can build solution.", "[heuristics][construction][RandomInsertion][init]") {
  auto stream = create_sequential_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};
  vrp::test::createDepotTask(problem, tasks);
  auto context = Context{problem.getShadow(), tasks.getShadow(), {}};

  random_insertion<Delegate>{}(context, 0, 0);
  //
  //  auto solution = Solution(std::move(problem), std::move(tasks));
  //  REQUIRE(SolutionChecker::check(solution).isValid());
}
