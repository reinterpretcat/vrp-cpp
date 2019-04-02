#pragma once

#include "Solver.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"

#include <nlohmann/json.hpp>

namespace vrp::test::here {
const auto SolverInstance = Solver<vrp::algorithms::refinement::create_refinement_context<>,
                                   vrp::algorithms::refinement::select_best_solution,
                                   vrp::algorithms::refinement::ruin_and_recreate_solution<>,
                                   vrp::algorithms::refinement::GreedyAcceptance<>,
                                   vrp::algorithms::refinement::MaxIterationCriteria,
                                   vrp::algorithms::refinement::log_to_console>{};

const auto DefaultTimeStart = "1970-01-01T00:00:00Z";
const auto SmallTimeEnd = "1970-01-01T00:01:40Z";
const auto LargeTimeEnd = "1970-01-01T00:16:40Z";

const auto SmallTimeWindows = nlohmann::json::array({nlohmann::json::array({DefaultTimeStart, SmallTimeEnd})});
const auto LargeTimeWindows = nlohmann::json::array({nlohmann::json::array({DefaultTimeStart, LargeTimeEnd})});
}