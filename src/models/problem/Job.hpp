#pragma once

#include "models/common/Duration.hpp"
#include "models/common/TimeWindow.hpp"
#include "models/problem/JobVisitor.hpp"

#include <string>
#include <vector>

namespace vrp::models::problem {
struct Job {
  /// Job id.
  std::string id;

  virtual void accept(JobVisitor&) const = 0;

  virtual ~Job() = default;
};
}  // namespace vrp::models::problem
