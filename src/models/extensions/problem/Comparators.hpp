#pragma once

#include "models/extensions/problem/Properties.hpp"

#include <memory>

namespace vrp::models::problem {

/// Compares jobs.
struct compare_jobs final {
  bool operator()(const problem::Job& lhs, const problem::Job& rhs) const {
    const static auto getter = get_job_id{};
    return getter(lhs) < getter(rhs);
  }
};

}  // namespace vrp::models::problem
