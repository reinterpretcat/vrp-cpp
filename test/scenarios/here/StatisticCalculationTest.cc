#include "Solver.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "streams/in/json/HereJson.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/streams/HereModelBuilders.hpp"

#include <any>
#include <catch/catch.hpp>

using namespace nlohmann;
using namespace vrp;
using namespace vrp::models::problem;
using namespace vrp::streams::in;
using namespace vrp::algorithms::refinement;

namespace {
auto solver = Solver<create_refinement_context<>,
                     select_best_solution,
                     ruin_and_recreate_solution<>,
                     GreedyAcceptance<>,
                     MaxIterationCriteria,
                     log_to_console>{};
}

namespace vrp::test::here {

SCENARIO("statistic can be calculated for two simple tours", "[scenario][statistic]") {
  GIVEN("problem with 3 jobs and two vehicle types") {
    WHEN("solve problem") {
      THEN("has no unassigned") {
        // TODO
      }

      THEN("has two tours") {
        // TODO
      }

      THEN("has total statistic") {
        // TODO
      }

      THEN("has first tour statistic") {
        // TODO
      }

      THEN("has second tour statistic") {
        // TODO
      }
    }
  }
}
}
