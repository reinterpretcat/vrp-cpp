#include "Solver.hpp"

#include "streams/in/LiLim.hpp"
#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/refinement/LogAndValidate.hpp"
#include "test_utils/streams/LiLimStreams.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms;
using namespace vrp::algorithms::construction;
using namespace vrp::streams::in;

namespace vrp::test {

SCENARIO("Solver can solve C101 problem greedy acceptance and default RaR", "[solver][default]") {
  auto solver = Solver<algorithms::refinement::create_refinement_context<>,
                       algorithms::refinement::select_best_solution,
                       algorithms::refinement::ruin_and_recreate_solution<>,
                       algorithms::refinement::GreedyAcceptance<>,
                       algorithms::refinement::MaxIterationCriteria,
                       vrp::test::log_and_validate>{};

  GIVEN("C101 problem with 25 customers") {
    auto stream = create_c101_25_problem_stream{}();
    auto problem = read_solomon_type<cartesian_distance>{}.operator()(stream);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has valid solution") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.first->routes.size() == 3);
      }
    }
  }
}

SCENARIO("Solver can solve LC101 problem greedy acceptance and default RaR", "[solver][default]") {
  auto solver = Solver<algorithms::refinement::create_refinement_context<>,
                       algorithms::refinement::select_best_solution,
                       algorithms::refinement::ruin_and_recreate_solution<>,
                       algorithms::refinement::GreedyAcceptance<>,
                       algorithms::refinement::MaxIterationCriteria,
                       algorithms::refinement::log_to_console>{};

  GIVEN("LC101 problem with 53 sequences") {
    auto stream = create_lc101_problem_stream{}();
    auto problem = read_li_lim_type<cartesian_distance>{}.operator()(stream);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has valid solution") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.first->routes.size() == 10);
        REQUIRE(std::abs(estimatedSolution.second.total() - 828.937) < 0.001);
      }
    }
  }
}
}
