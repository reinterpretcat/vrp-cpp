#pragma once

#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Activity.hpp"

#include <range/v3/all.hpp>
#include <set>
#include <vector>

namespace vrp::models::solution {

/// Represents a tour, a smart container for jobs with their associated activities.
class Tour final {
public:
  using Activity = std::shared_ptr<solution::Activity>;

  /// Adds activity within its job to end of tour.
  Tour& add(const Tour::Activity& activity) {
    insert(activity, activities_.size());
    return *this;
  }

  /// Inserts activity within its job at specified index.
  Tour& insert(const Tour::Activity& activity, size_t index) {
    activities_.insert(activities_.begin() + index, activity);

    if (activity->job.has_value()) { jobs_.insert(activity->job.value()); }

    return *this;
  }

  /// Removes job within its activities from the tour.
  Tour& remove(const solution::Activity::Job& job) {
    size_t removed = jobs_.erase(job);
    assert(removed == 1);

    ranges::action::remove_if(activities_, [&](const auto& a) { return a->job == job; });

    return *this;
  }

  /// Returns range view of all activities.
  auto activities() const { return ranges::view::all(activities_); }

  /// Returns range view of all jobs.
  auto jobs() const { return ranges::view::all(jobs_); }

private:
  /// Stores activities in the order the performed.
  std::vector<Tour::Activity> activities_;

  /// Stores jobs in the order of their activities added.
  std::set<solution::Activity::Job, problem::compare_jobs> jobs_;
};

}  // namespace vrp::models::solution
