#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <optional>

namespace vrp::models::solution {

/// Represents activity which is needed to be performed.
struct Activity final {
  /// Location where stop is performed.
  common::Location location;

  /// Specifies stop's schedule: actual arrival and departure time.
  common::Schedule schedule;

  /// Specifies activity's time window: an interval when job is allowed to be started.
  common::TimeWindow time;

  /// Specifies job relation. Empty if it has no relation to job.
  std::optional<problem::Job> job;
};

}  // namespace vrp::models::solution
