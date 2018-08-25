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

const auto T = Plan::assign();
const auto F = Plan::empty();

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

SCENARIO("Can mutate specific individuum.", "[genetic][mutation][c101][specific]") {
  auto solution = createPopulation(
    createProblem<create_c101_problem_stream>(),

    {0, 20, 21, 22, 23, 24, 25, 10, 11, 9, 6, 4, 5, 3, 7, 8, 2, 1, 13, 17, 18, 19, 15, 16, 14, 12,
     0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0},

    {0,          10,         12,         12.1655254, 13,         15,         17,         34.2046509,
     37.2046509, 40.3669281, 42.6029968, 44.8390656, 15.1327457, 16.1327457, 18.1327457, 20.9611721,
     28.2412815, 30.2412815, 30.8058434, 34.8058434, 37.8058434, 42.8058434, 47.8058434, 52.8058434,
     54.8058434, 57.8058434, 0,          0,          0,          0,          0,          0,
     0,          0,          0,          0,          0,          0,          0,          0,
     0,          0,          0,          0,          0,          0,          0,          0,
     0,          0,          0,          0},

    {0,   100, 1004, 902, 822, 155, 259, 447, 540, 633, 725, 817, 105, 196, 288, 380, 915, 1007,
     120, 214, 307,  402, 497, 592, 684, 777, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

    {200, 190, 170, 180, 190, 190, 150, 140, 130, 120, 100, 90, 190, 180, 160, 140, 110, 100,
     170, 150, 130, 120, 80,  40,  30,  10,  200, 0,   0,   0,  0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},

    {0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

    {T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T,
     T, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F});
  auto mutation = Mutation{0, 1, {0.880393744, 2}};

  thrust::for_each(exec_unit, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
                   run_mutation{solution.getShadow(), mutation});
  
  REQUIRE(SolutionChecker::check(solution).isValid());
}