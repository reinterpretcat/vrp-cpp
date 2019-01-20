#include "Solver.hpp"

#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/refinement/LogAndValidate.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>
#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::algorithms::construction;
using namespace vrp::streams::in;

namespace vrp::test {

SCENARIO("Solver can solve c101 problem greedy acceptance and default RaR", "[solver][default]") {
  auto solver = Solver<algorithms::refinement::create_refinement_context<>,
                       algorithms::refinement::select_best_solution,
                       algorithms::refinement::ruin_and_recreate_solution<>,
                       algorithms::refinement::GreedyAcceptance<>,
                       algorithms::refinement::MaxIterationCriteria,
                       vrp::test::log_and_validate>{};

  GIVEN("C101 problem with 25 customers") {
    auto stream =
      std::fstream("/home/builuk/playground/vrp/resources/data/solomon/benchmarks/c101.100.txt", std::ios::in);
    auto problem = read_solomon_type<cartesian_distance>{}.operator()(stream);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has valid solution") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        // REQUIRE(estimatedSolution.first->routes.size() == 3);
      }
    }
  }
}
}
