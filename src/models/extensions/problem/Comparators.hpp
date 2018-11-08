#pragma once

#include "models/problem/Job.hpp"

namespace vrp::models::problem {

/// Compares jobs.
struct compare_jobs final {
  bool operator()(const problem::Job& lhs, const problem::Job& rhs) const {
    return lhs.id < rhs.id;
  }
};

}  // namespace vrp::models::problem
