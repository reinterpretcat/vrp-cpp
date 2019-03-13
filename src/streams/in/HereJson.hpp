#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/costs/MatrixTransportCosts.hpp"

#include <istream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>

namespace vrp::streams::in {
// TODO
}