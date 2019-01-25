#pragma once

#include "models/common/Dimension.hpp"
#include "models/problem/JobDetail.hpp"

#include <string>
#include <vector>

namespace vrp::models::problem {

/// Represents a job which should be performed once but actual place/time might vary.
struct Service final {
  /// Specifies service details: where and when it can be performed.
  std::vector<JobDetail> details;

  /// Dimensions which simulates work requirements.
  common::Dimensions dimens;
};

}  // namespace vrp::models::problem
