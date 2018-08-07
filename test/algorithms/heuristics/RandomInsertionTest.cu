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
inline void test_heuristic(int populationSize,
                           vector_ptr<Convolution> convolutions,
                           std::function<void(Problem&, Tasks&)> modificator,
                           std::function<void(const Solution&)> validator) {
  auto stream = ProblemStream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  Tasks tasks{problem.size(), problem.size() * populationSize};

  modificator(problem, tasks);

  thrust::for_each_n(exec_unit, thrust::make_counting_iterator(0), 1,
                     run_heuristic{Context{problem.getShadow(), tasks.getShadow(), convolutions}});

  auto solution = Solution(std::move(problem), std::move(tasks));

  validator(solution);
}

template<typename ProblemStream>
inline void test_individuum() {
  test_heuristic<ProblemStream>(
    1, {}, [](Problem& problem, Tasks& tasks) { vrp::test::createDepotTask(problem, tasks); },
    [](const Solution& solution) {
      MatrixTextWriter::write(std::cout, solution);
      REQUIRE(SolutionChecker::check(solution).isValid());
    });
}

template<typename ProblemStream>
inline void test_population() {
  auto stream = ProblemStream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());

  auto tasks = create_population<random_insertion<TransitionOperator>>{problem}(2);

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

SCENARIO("Can create solution using convolution from another solution.",
         "[heuristics][construction][NearestNeighbor][convolutions]") {
  int size = 6;
  auto convolutions =
    vrp::test::create<Convolution>({Convolution{size, 2, 30, {3, 2}, {0, 22}, {4, 5}}});
  auto modificator = [=](Problem& problem, Tasks& tasks) {
    vrp::test::createDepotTask(problem, tasks);
    tasks.plan[2] = Plan::reserve(0);
    tasks.plan[3] = Plan::reserve(0);

    auto begin = tasks.ids.begin() + size;
    thrust::sequence(exec_unit, begin + 4, begin + size, 3, -1);
  };
  auto validator = [=](const Solution& solution) {
    CHECK_THAT(vrp::test::copy(solution.tasks.ids, size),
               Catch::Matchers::Equals(std::vector<int>{0, 1, 3, 2, 4, 5}));
    CHECK_THAT(vrp::test::copy(solution.tasks.vehicles, size),
               Catch::Matchers::Equals(std::vector<int>{0, 0, 0, 0, 0, 0}));
    CHECK_THAT(vrp::test::copy(solution.tasks.capacities, size),
               Catch::Matchers::Equals(std::vector<int>{10, 9, 8, 7, 6, 5}));
    CHECK_THAT(vrp::test::copy(solution.tasks.times, size),
               Catch::Matchers::Equals(std::vector<int>{0, 11, 23, 34, 46, 57}));
    CHECK_THAT(vrp::test::copy(solution.tasks.costs, size),
               Catch::Matchers::Equals(std::vector<float>{0, 1, 3, 4, 6, 7}));
  };

  test_heuristic<create_sequential_problem_stream>(2, convolutions.data(), modificator, validator);
}
