#pragma once

#include "models/common/Location.hpp"
#include "models/common/Dimension.hpp"
#include "Job.hpp"

#include <optional>

namespace vrp::models::problem {

/// Represents a job associated with single location.
struct Service final : public Job {

  /// Specifies location where service has to be performed.
  std::optional<vrp::models::common::Location> location;

  /// Time has to be spend performing job.
  vrp::models::common::Duration duration;

  /// Time windows when job can be performed.
  std::vector<vrp::models::common::TimeWindow> times;

  /// Dimensions which simulates work requirements.
  vrp::models::common::Dimensions dimensions;

  void accept(JobVisitor &visitor) const override {
    visitor.visit(*this);
  }
};

}
