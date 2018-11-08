#pragma once

#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Activity.hpp"

#include <map>
#include <memory>
#include <set>

namespace vrp::models::solution {

/// Represents a tour, a smart container for jobs with their associated activities.
class Tour final {
 public:
  /// Adds activity within its job to the tour.
  Tour& add(const solution::Activity& activity) { return *this; }

  /// Removes job within its activities from the tour.
  Tour& remove(std::shared_ptr<const problem::Job>& job) { return *this; }

private:
  /// Stores activities in the order the performed.
  std::set<solution::Activity, solution::compare_activities> activities;

  /// Stores job -> activities relations.
  std::
    map<std::shared_ptr<const problem::Job>, std::vector<solution::Activity>, problem::compare_jobs>
      relations;
};

}  // namespace vrp::models::solution