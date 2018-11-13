#pragma once

#include "models/common/Cost.hpp"

namespace vrp::algorithms::construction {

/// Provides the way to get some meta information about insertion progress.
struct InsertionProgress final {
  /// Specifies best known cost.
  models::common::Cost cost;

  /// Specifies solution completeness.
  double completeness;
};
}