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

}  // namespace

SCENARIO("Can mutate c101 individuum keeping convolutions.", "[genetic][mutation]") {
  auto stream = create_c101_problem_stream{}();
  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 2);
  auto mutation = Mutation{0, 1, 0, false};

  thrust::for_each(exec_unit, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
                   run_mutation{solution.getShadow(), mutation});

  MatrixTextWriter::write(std::cout, solution);
  REQUIRE(SolutionChecker::check(solution).isValid());
}