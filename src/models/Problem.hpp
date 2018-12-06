#pragma once

#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <set>

namespace vrp::models {

/// Defines VRP problem.
struct Problem final {
  std::shared_ptr<problem::Fleet> fleet;

  std::set<models::problem::Job, models::problem::compare_jobs> jobs;
};
}
