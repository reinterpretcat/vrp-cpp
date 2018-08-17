#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "config.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/PopulationFactory.hpp"
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
struct run_mutation final {
  Solution::Shadow solution;
  const Mutation mutation;

  EXEC_UNIT void operator()(int index) { create_mutant<TransitionOperator>{solution}(mutation); }
};

template<typename Problem, int Size>
void test(const Mutation& mutation) {
  auto stream = Problem{}();
  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 2);
  // NOTE invalidate ids for test purpose only
  auto start = Size + 1 + 1;
  thrust::fill(exec_unit, solution.tasks.ids.begin() + start, solution.tasks.ids.end(), -1);
  thrust::fill(exec_unit, solution.tasks.vehicles.begin() + start, solution.tasks.vehicles.end(),
               -1);

  thrust::for_each(exec_unit, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
                   run_mutation{solution.getShadow(), mutation});

  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}

}  // namespace

SCENARIO("Can mutate c101 individuum keeping convolutions.", "[genetic][mutation][c101]") {
  test<create_c101_problem_stream, 25>(Mutation{0, 1, {0.75, 2}});
}

SCENARIO("Can mutate rc1_10_1 individuum keeping convolutions.", "[genetic][mutation][rc1_10_1]") {
  test<rc1_10_1_problem_stream, 1000>(Mutation{0, 1, {0.75, 3}});
}