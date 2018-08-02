#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/genetic/Populations.hpp"
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
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::utils;
using namespace vrp::test;

namespace {

struct run_heuristic final {
  Context context;
  EXEC_UNIT void operator()(int index) const {
    random_insertion<TransitionOperator>{}(context, index, 0);
  };
};

template<typename ProblemStream>
void test() {
  auto stream = ProblemStream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};
  vrp::test::createDepotTask(problem, tasks);

  thrust::for_each_n(exec_unit, thrust::make_counting_iterator(0), 1,
                     run_heuristic{Context{problem.getShadow(), tasks.getShadow(), {}}});

  auto solution = Solution(std::move(problem), std::move(tasks));
  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}

}  // namespace

SCENARIO("Can build single solution with one vehicle.",
         "[heuristics][construction][RandomInsertion]") {
  test<create_sequential_problem_stream>();
}

SCENARIO("Can build single solution with multiple vehicles.",
         "[heuristics][construction][RandomInsertion]") {
  test<create_exceeded_time_problem_stream>();
}

SCENARIO("Can build single solution with C101 problem.",
         "[heuristics][construction][RandomInsertion]") {
  test<create_c101_problem_stream>();
}

SCENARIO("Can build population with one vehicle in each of two individuums.",
         "[heuristics][construction][RandomInsertion]") {
  auto stream = create_sequential_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  auto settings = Settings{2, vrp::algorithms::convolutions::Settings{0, 0}};

  auto tasks = create_population<random_insertion<TransitionOperator>>(problem)(settings);

  auto solution = Solution(std::move(problem), std::move(tasks));
  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}