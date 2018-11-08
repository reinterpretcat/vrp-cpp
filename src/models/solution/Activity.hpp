#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <optional>

namespace vrp::models::solution {

/// Represents activity which is needed to be performed.
struct Activity final {
  using Job = std::shared_ptr<const problem::Job>;

  /// Specifies activity schedule.
  vrp::models::common::Schedule schedule;

  /// Location where activity is performed.
  vrp::models::common::Location location;

  /// Specifies job relation. Empty if it has no relation to job.
  std::optional<Activity::Job> job;
};

}  // namespace vrp::models::solution
