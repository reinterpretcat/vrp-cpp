#pragma once

#include "models/problem/Job.hpp"

#include <memory>

namespace vrp::models::problem {

/// Compares jobs.
struct compare_jobs final {
  bool operator()(const problem::Job& lhs, const problem::Job& rhs) const { return lhs.id < rhs.id; }

  bool operator()(const std::shared_ptr<const problem::Job>& lhs,
                  const std::shared_ptr<const problem::Job>& rhs) const {
    return this->operator()(*lhs, *rhs);
  }
};

}  // namespace vrp::models::problem
