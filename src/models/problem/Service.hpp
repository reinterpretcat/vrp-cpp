#pragma once

#include "models/common/Dimension.hpp"
#include "models/problem/Detail.hpp"

#include <vector>

namespace vrp::models::problem {

/// Represents a job which should be performed once but actual place/time might vary.
struct Service final {
  /// Specifies service id.
  std::string id;

  /// Specifies service details: where and when it can be performed.
  std::vector<Detail> details;

  /// Dimensions which simulates work requirements.
  common::Dimensions dimens;
};

}  // namespace vrp::models::problem
