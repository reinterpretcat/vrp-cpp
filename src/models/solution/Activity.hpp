#pragma once

#include "models/problem/Job.hpp"
#include "models/solution/Stop.hpp"

#include <memory>
#include <optional>

namespace vrp::models::solution {

/// Represents activity which is needed to be performed.
struct Activity final {
  /// Specifies activity's stop.
  solution::Stop stop;

  /// Specifies activity's time window: an interval when job is allowed to be started.
  common::TimeWindow interval;

  /// Specifies job relation. Empty if it has no relation to job.
  std::optional<problem::Job> job;
};

}  // namespace vrp::models::solution
