#pragma once

#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/solution/Helpers.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Activity.hpp"

#include <algorithm>
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

    // TODO activity inserted in tour should have always job?
    auto job = retrieve_job{}(*activity);
    if (job.has_value()) { jobs_.insert(job.value()); }

    return *this;
  }

  /// Removes job within its activities from the tour.
  Tour& remove(const problem::Job& job) {
    size_t removed = jobs_.erase(job);
    assert(removed == 1);

    ranges::action::remove_if(activities_, [&](const auto& a) { return retrieve_job{}(*a) == job; });

    return *this;
  }

  /// Returns range view of all activities.
  auto activities() const { return ranges::view::all(activities_); }

  /// Returns range view of all activities for specific view.
  auto activities(const problem::Job& job) const {
    return ranges::view::all(activities_) |
      ranges::view::remove_if([&](const auto& a) { return retrieve_job{}(*a) != job; });
  }

  /// Returns range view of all jobs.
  auto jobs() const { return ranges::view::all(jobs_); }

  /// Returns activity by its index in tour.
  Activity get(size_t index) const { return activities_[index]; }

  /// Returns first activity in tour.
  Activity first() const { return activities_.front(); }

  /// Returns last activity in tour.
  Activity last() const { return activities_.back(); }

  /// Returns index of first job occurrence in the tour.
  /// Throws exception if job is not present.
  /// Complexity: O(n)
  std::size_t index(const problem::Job& job) const {
    auto i =
      std::find_if(activities_.begin(), activities_.end(), [&](const auto& a) { return retrieve_job{}(*a) == job; });

    if (i == activities_.end()) throw std::invalid_argument("Cannot find job");

    return std::distance(activities_.begin(), i);
  }

  /// Checks whether tour is empty.
  bool empty() const { return activities_.empty(); }

  /// Checks whether job is present in tour.
  bool has(const problem::Job& job) const { return jobs_.find(job) != jobs_.end(); }

  /// Returns separately amount of jobs and activities in tour.
  std::pair<std::size_t, std::size_t> sizes() const { return std::make_pair(jobs_.size(), activities_.size()); }

private:
  /// Stores activities in the order the performed.
  std::vector<Tour::Activity> activities_;

  /// Stores jobs in the order of their activities added.
  std::set<problem::Job, problem::compare_jobs> jobs_;
};

}  // namespace vrp::models::solution
