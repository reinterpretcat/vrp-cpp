#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "config.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SpecificSolutions.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;
using namespace vrp::utils;

namespace {

/// Runs crossover
template <typename Mutator>
struct run_mutation final {
  Solution::Shadow solution;
  const Mutation mutation;

  EXEC_UNIT void operator()(int index) {
    Mutator{solution}(mutation);
  }
};

template<typename Mutator, typename ProblemDesc, int Size>
void test_with_problem(const Mutation& mutation) {
  auto stream = ProblemDesc{}();
  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 2);
  // NOTE invalidate ids for test purpose only
  auto start = Size + 1 + 1;
  thrust::fill(exec_unit, solution.tasks.ids.begin() + start, solution.tasks.ids.end(), -1);
  thrust::fill(exec_unit, solution.tasks.vehicles.begin() + start, solution.tasks.vehicles.end(),
               -1);

  thrust::for_each(exec_unit, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
                   run_mutation<Mutator>{solution.getShadow(), mutation});

  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}

template <typename Mutator, typename SolutionDesc>
Solution test_with_solution(const Mutation& mutation) {
  auto solution = SolutionDesc{}();

  thrust::for_each(exec_unit, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
                   run_mutation<Mutator>{solution.getShadow(), mutation});

  REQUIRE(SolutionChecker::check(solution).isValid());
  return std::move(solution);
}

}  // namespace

SCENARIO("Can mutate c101 individuum keeping convolutions.", "[genetic][mutation][weak_subtours][c101]") {
  test_with_problem<mutate_weak_subtours<TransitionOperator>, create_c101_problem_stream, 25>(Mutation{0, 1, {0.75, 2}});
}

SCENARIO("Can mutate rc1_10_1 individuum keeping convolutions.", "[genetic][mutation][weak_subtours][rc1_10_1]") {
  test_with_problem<mutate_weak_subtours<TransitionOperator>, rc1_10_1_problem_stream, 1000>(Mutation{0, 1, {0.75, 3}});
}

SCENARIO("Can mutate specific individuum.", "[genetic][mutation][c101][weak_subtours][specific]") {
  test_with_solution<mutate_weak_subtours<TransitionOperator>, create_c101_specific_individuum_1>({0, 1, {0.880393744, 2}});
}

SCENARIO("Can escape local minimum.", "[genetic][mutation][c101][weak_subtours][specific]") {
  auto solution = test_with_solution<mutate_weak_subtours<TransitionOperator>, create_c101_near_optimum>({0, 1, {1, 2}});
  REQUIRE(solution.tasks.vehicles.back() == 2);
}
