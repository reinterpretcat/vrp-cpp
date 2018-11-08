#pragma once

#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Activity.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <set>
#include <vector>

namespace vrp::models::solution {

/// Represents a tour, a smart container for jobs with their associated activities.
class Tour final {
public:
  /// Adds activity within its job to end of tour.
  Tour& add(const solution::Activity& activity) {
    insert(activity, activities_.size());
    return *this;
  }

  /// Inserts activity within its job at specified index.
  Tour& insert(const solution::Activity& activity, size_t index) {
    activities_.insert(activities_.begin() + index, activity);

    if (activity.job.has_value()) { jobs_.insert(activity.job.value()); }

    return *this;
  }

  /// Removes job within its activities from the tour.
  Tour& remove(std::shared_ptr<const problem::Job>& job) { return *this; }

  auto activities() const { return ranges::v3::view::all(activities_); }

  auto jobs() const { return ranges::v3::view::all(jobs_); }

private:
  /// Stores activities in the order the performed.
  std::vector<solution::Activity> activities_;

  /// Stores jobs in the order of their activities added.
  std::set<Activity::Job, problem::compare_jobs> jobs_;
};

}  // namespace vrp::models::solution
