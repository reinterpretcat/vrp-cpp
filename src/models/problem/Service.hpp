#pragma once

#include "models/common/Dimension.hpp"
#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/TimeWindow.hpp"

#include <optional>
#include <string>
#include <vector>

namespace vrp::models::problem {

/// Represents a job which should be performed once but actual place/time might vary.
struct Service final {
  /// Represents a work which has to be performed.
  struct Detail final {
    /// Location where work has to be performed.
    std::optional<common::Location> location;

    /// Time has to be spend performing work.
    common::Duration duration;

    /// Time windows when work can be started.
    std::vector<common::TimeWindow> times;
  };

  /// Specifies service details: where and when it can be performed.
  std::vector<Detail> details;

  /// Dimensions which simulates work requirements.
  common::Dimensions dimens;
};

}  // namespace vrp::models::problem
