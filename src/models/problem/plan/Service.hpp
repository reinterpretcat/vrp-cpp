#pragma once

#include "models/common/Location.hpp"
#include "models/common/Dimension.hpp"
#include "models/problem/plan/Job.hpp"

#include <optional>

namespace vrp::models::problem::plan {

/// Represents a job associated with single location.
struct Service final : public Job {

  vrp::models::common::Location location;

  vrp::models::common::Dimensions dimensions;

  void accept(JobVisitor &visitor) const override {
    visitor.visit(*this);
  }
};

}
