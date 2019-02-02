#pragma once

#include "models/common/Distance.hpp"

#include <cmath>
#include <utility>

namespace vrp::streams::in {

/// Calculates cartesian distance between two points on plane in 2D.
struct cartesian_distance final {
  models::common::Distance operator()(const std::pair<int, int>& left, const std::pair<int, int>& right) {
    models::common::Distance x = left.first - right.first;
    models::common::Distance y = left.second - right.second;
    return std::sqrt(x * x + y * y);
  }
};
}
