#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "config.hpp"
#include "streams/input/SolomonReader.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <catch/catch.hpp>
#include <functional>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::genetic;
using namespace vrp::runtime;
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
inline void test_individuum(vector_ptr<Convolution> convolutions,
    std::function<void(Problem&, Tasks&)> modificator) {
  auto stream = ProblemStream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size()};

  modificator(problem, tasks);

  thrust::for_each_n(exec_unit, thrust::make_counting_iterator(0), 1,
                     run_heuristic{Context{problem.getShadow(), tasks.getShadow(), convolutions}});

  auto solution = Solution(std::move(problem), std::move(tasks));
  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}

template<typename ProblemStream>
inline void test_individuum() {
  test_individuum<ProblemStream>({}, [](Problem& problem, Tasks& tasks){
    vrp::test::createDepotTask(problem, tasks);
  });
}

template<typename ProblemStream>
inline void test_population() {
  auto stream = ProblemStream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  auto settings = Settings{2, vrp::algorithms::convolutions::Settings{0, 0}};

  auto tasks = create_population<random_insertion<TransitionOperator>>(problem)(settings);

  auto solution = Solution(std::move(problem), std::move(tasks));
  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}

}  // namespace

SCENARIO("Can build single solution with one vehicle.",
         "[heuristics][construction][RandomInsertion]") {
  test_individuum<create_sequential_problem_stream>();
}

SCENARIO("Can build single solution with multiple vehicles.",
         "[heuristics][construction][RandomInsertion]") {
  test_individuum<create_exceeded_time_problem_stream>();
}

SCENARIO("Can build single solution with C101 problem.",
         "[heuristics][construction][RandomInsertion]") {
  test_individuum<create_c101_problem_stream>();
}

SCENARIO("Can build population with one vehicle in each of two individuums.",
         "[heuristics][construction][RandomInsertion]") {
  test_population<create_sequential_problem_stream>();
}

SCENARIO("Can build population with C101 problem and two individuums.",
         "[heuristics][construction][RandomInsertion]") {
  test_population<create_c101_problem_stream>();
}

SCENARIO("Can create single solution with convolution.",
         "[ggg][heuristics][construction][NearestNeighbor][convolutions]") {
  auto convolutions = vrp::test::create<Convolution>({Convolution{0, 3, 30, {1, 3}, {0, 33}, {3, 5}} });
  auto modificator = [](Problem& problem, Tasks& tasks){
    vrp::test::createDepotTask(problem, tasks);
    thrust::sequence(exec_unit, tasks.ids.begin() + 3, tasks.ids.end(), 1);
    thrust::fill(exec_unit, tasks.plan.begin() + 3, tasks.plan.end(), Plan::reserve(0));
  };

  test_individuum<create_sequential_problem_stream>(convolutions.data(), modificator);
}
