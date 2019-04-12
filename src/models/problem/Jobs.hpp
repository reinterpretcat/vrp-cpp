#pragma once

#include "models/common/Timestamp.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/problem/Distances.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Job.hpp"

#include <limits>
#include <map>
#include <pstl/algorithm>
#include <pstl/execution>
#include <range/v3/all.hpp>
#include <set>
#include <vector>

namespace vrp::models::problem {

/// Calculates job neighborhood in terms of the cost.
struct Jobs final {
  Jobs(const costs::TransportCosts& transport, const Fleet& fleet, ranges::any_view<Job> jobs) {
    createJobs(jobs);
    createJobIndex(transport, fleet);
  };

  /// Allow only move.
  Jobs(Jobs&& other) noexcept : jobs_(std::move(other.jobs_)), index_(std::move(other.index_)) {}
  Jobs(const Jobs&) = delete;
  Jobs& operator=(const Jobs&) = delete;

  /// Returns all jobs.
  ranges::any_view<Job> all() const { return ranges::view::all(jobs_); }

  /// Returns all jobs with distance zero to given.
  ranges::any_view<Job> ghosts(const common::Profile profile, const Job& job, const common::Timestamp time) const {
    return index_.find(profile)->second.find(job)->second.first |
      ranges::view::remove_if([=](const auto& pair) { return pair.second == 0; }) |
      ranges::view::transform([](const auto& pair) { return pair.first; });
  }

  /// Returns range of jobs "near" to given one applying filter predicate on "near" value.
  /// Near is defined by transport costs, its profile and time. Value is filtered by max distance.
  ranges::any_view<Job> neighbors(const common::Profile profile,
                                  const Job& job,
                                  const common::Timestamp time,
                                  common::Distance maxDistance = std::numeric_limits<common::Distance>::max()) const {
    return index_.find(profile)->second.find(job)->second.first |
      ranges::view::remove_if([=](const auto& pair) { return pair.second == 0 || pair.second > maxDistance; }) |
      ranges::view::transform([](const auto& pair) { return pair.first; });
  }

  /// Returns job rank as distance to any vehicle's start position.
  common::Distance rank(common::Profile profile, const Job& job) const {
    return index_.find(profile)->second.find(job)->second.second;
  }

  /// Returns amount of all jobs.
  std::size_t size() const { return jobs_.size(); }

private:
  void createJobs(ranges::any_view<Job>& jobs) { ranges::copy(jobs, ranges::back_inserter(jobs_)); }

  /// Creates time independent job index for each profile.
  void createJobIndex(const costs::TransportCosts& transport, const Fleet& fleet) {
    using namespace ranges;

    ranges::for_each(fleet.profiles(), [&](const auto& profile) {
      auto& map = index_[profile];
      auto distance = models::problem::job_distance{transport, profile, common::Timestamp{}};

      // get all possible start positions for given profile
      auto starts =
        view::for_each(fleet.vehicles() | view::remove_if([&](const auto& v) { return v->profile != profile; }),
                       [](const auto& v) {
                         return view::for_each(v->details,
                                               [](const auto& detail) { return ranges::yield(detail.start); });
                       }) |
        to_vector;

      // preinsert all keys
      std::transform(jobs_.begin(), jobs_.end(), std::inserter(map, map.end()), [&](const auto& job) {
        return std::make_pair(job, JobIndex{});
      });

      // process all values in parallel
      std::for_each(pstl::execution::par, jobs_.begin(), jobs_.end(), [&](const auto& job) {
        auto jobDistances = jobs_ | view::remove_if([&](const auto& j) { return job == j; }) |
          view::transform([&](const auto& j) { return std::make_pair(j, distance(j, job)); }) | ranges::to_vector |
          action::sort([](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });

        auto fleetDistance = ranges::min(starts | view::transform([&](const auto& s) { return distance(job, s); }));

        map[job] = std::pair(std::move(jobDistances), fleetDistance);
      });
    });
  }

  using JobIndex = std::pair<std::vector<std::pair<Job, common::Distance>>, common::Distance>;

  std::vector<Job> jobs_;
  std::map<common::Profile, std::map<Job, JobIndex, compare_jobs>> index_;
};
}