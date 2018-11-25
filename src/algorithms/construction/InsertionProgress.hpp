#pragma once

#include "models/common/Cost.hpp"

namespace vrp::algorithms::construction {

/// Provides the way to get some meta information about insertion progress.
struct InsertionProgress final {
  /// Specifies best known cost depending on context.
  models::common::Cost bestCost;

  /// Specifies solution completeness.
  double completeness;
};
}