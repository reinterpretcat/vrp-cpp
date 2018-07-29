#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "config.hpp"
#include "streams/input/SolomonReader.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <catch/catch.hpp>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

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

struct run_heuristic final {
  Context context;
  EXEC_UNIT void operator()(int index) const { random_insertion<Delegate>{}(context, index, 0); };
};

}  // namespace

SCENARIO("Can build single solution.", "[heuristics][construction][RandomInsertion][init]") {
  auto stream = create_sequential_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};
  vrp::test::createDepotTask(problem, tasks);

  thrust::for_each_n(exec_unit, thrust::make_counting_iterator(0), 1,
                     run_heuristic{Context{problem.getShadow(), tasks.getShadow(), {}}});

  auto solution = Solution(std::move(problem), std::move(tasks));
  // MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}
