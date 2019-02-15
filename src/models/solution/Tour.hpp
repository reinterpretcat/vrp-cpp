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

  /// Sets tour start.
  Tour& start(const Tour::Activity& activity) {
    assert(activities_.empty());
    assert(!activity->service.has_value());
    activities_.push_back(activity);
    return *this;
  }

  /// Sets tour end.
  Tour& end(const Tour::Activity& activity) {
    assert(!activities_.empty());
    assert(!activity->service.has_value());
    isClosed_ = true;
    activities_.push_back(activity);
    return *this;
  }

  /// Inserts activity within its job to the end of tour.
  Tour& insert(const Tour::Activity& activity) {
    insert(activity, count() + 1);
    return *this;
  }

  /// Inserts activity within its job at specified index.
  Tour& insert(const Tour::Activity& activity, size_t index) {
    assert(activity->service.has_value());
    assert(!activities_.empty());

    activities_.insert(activities_.begin() + index, activity);

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

  /// Returns counted tour legs.
  auto legs() const {
    using namespace ranges;
    auto size = std::max<int>(2, static_cast<int>(activities_.size())) - 1;
    return view::zip(activities_ | view::cycle | view::sliding(2) | view::take(size),
                     view::iota(static_cast<size_t>(0)));
  }

  /// Returns range view of all jobs.
  auto jobs() const { return ranges::view::all(jobs_); }

  /// Returns activity by its index in tour.
  Activity get(size_t index) const { return activities_[index]; }

  /// Returns start activity in tour.
  Activity start() const { return activities_.front(); }

  /// Returns end activity in tour.
  Activity end() const { return activities_.back(); }

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

  /// Checks whether tour has jobs.
  bool hasJobs() const { return !jobs_.empty(); }

  /// Returns amount of job activities.
  std::size_t count() const { return empty() ? 0 : activities_.size() - (isClosed_ ? 2 : 1); }

  /// Creates a deep copy of existing tour.
  Tour copy() const {
    auto newTour = Tour{};
    newTour.activities_.reserve(activities_.size());
    newTour.isClosed_ = isClosed_;
    newTour.jobs_ = jobs_;

    ranges::for_each(activities_, [&](const auto& activity) {
      newTour.activities_.push_back(std::make_shared<solution::Activity>(solution::Activity{*activity}));
    });

    return std::move(newTour);
  }

private:
  /// Stores activities in the order the performed.
  std::vector<Tour::Activity> activities_;

  /// Stores jobs in the order of their activities added.
  std::set<problem::Job, problem::compare_jobs> jobs_;

  /// Keeps track whether tour is set as closed.
  bool isClosed_ = false;
};

}  // namespace vrp::models::solution
