#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"

#include <optional>

namespace vrp::algorithms::construction {

/// Checks whether vehicle can serve activity taking into account their time windows.
struct VehicleActivityTimeWindow final {
  std::optional<std::tuple<bool, int>> check(const InsertionRouteContext&, const InsertionActivityContext&) const {
    return {};
  }
};
}
