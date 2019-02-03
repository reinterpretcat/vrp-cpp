#pragma once

#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <optional>

namespace vrp::models::solution {

/// Represents activity which is needed to be performed.
struct Activity final {
  /// Specifies details of activity: a variant of job details.
  struct Detail final {
    /// Location where activity is performed.
    common::Location location;

    /// Specifies activity's duration.
    common::Duration duration;

    /// Specifies activity's time window: an interval when job is allowed to be started.
    common::TimeWindow time;
  };

  /// Specifies activity details.
  Detail detail;

  /// Specifies activity's schedule: actual arrival and departure time.
  common::Schedule schedule;

  /// Specifies service relation. Empty if it has no relation to service (e.g. tour start or end).
  /// If service is part of sequence, then original sequence can be received via its dimens.
  std::optional<std::shared_ptr<const problem::Service>> service;
};

}  // namespace vrp::models::solution
