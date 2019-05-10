#pragma once

#include "test_utils/Solvers.hpp"

#include <nlohmann/json.hpp>

namespace vrp::test::here {
const auto SolverInstance = create_default_solver<>{}();

const auto DefaultTimeStart = "1970-01-01T00:00:00Z";
const auto SmallTimeEnd = "1970-01-01T00:01:40Z";
const auto LargeTimeEnd = "1970-01-01T00:16:40Z";

const auto SmallTimeWindows = nlohmann::json::array({nlohmann::json::array({DefaultTimeStart, SmallTimeEnd})});
const auto LargeTimeWindows = nlohmann::json::array({nlohmann::json::array({DefaultTimeStart, LargeTimeEnd})});
}