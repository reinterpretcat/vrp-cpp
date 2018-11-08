#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <variant>

namespace vrp::models::solution {

/// Represents activity which is needed to be performed.
struct Activity final {
  /// Specifies activity schedule.
  vrp::models::common::Schedule schedule;

  /// Location where activity is performed.
  vrp::models::common::Location location;

  /// Specifies job relation. Empty if it has no relation to job.
  std::variant<std::monostate, std::shared_ptr<const vrp::models::problem::Job>> job;
};

}  // namespace vrp::models::solution
