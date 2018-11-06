#pragma once

#include "models/common/TimeWindow.hpp"
#include "models/common/Duration.hpp"
#include "models/problem/plan/JobVisitor.hpp"

#include <string>
#include <vector>

namespace vrp::models::problem::plan {
  struct Job {
    /// Job id.
    std::string id;

    /// Time has to be spend performing job.
    vrp::models::common::Duration duration;

    /// Time windows when job can be performed.
    std::vector<vrp::models::common::TimeWindow> times;

    virtual void accept(JobVisitor &) const = 0;

    virtual ~Job() = default;

  };
}
