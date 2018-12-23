#pragma once

#include "models/common/Timestamp.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/problem/Distances.hpp"
#include "models/problem/Job.hpp"

#include <map>
#include <pstl/algorithm>
#include <pstl/execution>
#include <range/v3/all.hpp>
#include <set>
#include <string>
#include <vector>

namespace vrp::models::problem {

/// Calculates job neighborhood in terms of the cost.
struct Jobs final {
  Jobs(const costs::TransportCosts& transport, ranges::any_view<Job> jobs, ranges::any_view<std::string> profiles) {
    createJobSet(jobs);
    createJobIndex(transport, profiles);
  };

  /// Allow only move.
  Jobs(Jobs&& other) noexcept : jobs_(std::move(other.jobs_)), index_(std::move(other.index_)) {}
  Jobs(const Jobs&) = delete;
  Jobs& operator=(const Jobs&) = delete;

  /// Returns all jobs.
  ranges::any_view<Job> all() const { return ranges::view::all(jobs_); }

  /// Returns range of jobs "near" to given one applying filter predicate on "near" value.
  /// Near is defined by transport costs and their profile and time which value is filtered by Filter.
  template<typename Filter>
  ranges::any_view<Job> neighbors(const std::string& profile,
                                  const Job& job,
                                  const common::Timestamp time,
                                  Filter filter) const {
    return index_.find(profile)->second.find(job)->second |
      ranges::view::remove_if([&filter](const auto& pair) { return !filter(pair.second); }) |
      ranges::view::transform([](const auto& pair) { return pair.first; });
  }

private:
  void createJobSet(ranges::any_view<Job>& jobs) { ranges::copy(jobs, ranges::inserter(jobs_, jobs_.begin())); }

  /// Creates time independent job index for each profile.
  void createJobIndex(const costs::TransportCosts& transport, ranges::any_view<std::string>& profiles) {
    using namespace ranges;

    ranges::for_each(profiles, [&](const auto& profile) {
      auto map = index_[profile];  // std::map<Job, std::vector<std::pair<Job, common::Distance>>, compare_jobs>{};
      auto distance = models::problem::job_distance{transport, profile, common::Timestamp{}};

      // preinsert all keys
      std::transform(jobs_.begin(), jobs_.end(), std::inserter(map, map.end()), [](const auto& j) {
        return std::make_pair(j, std::vector<std::pair<Job, common::Distance>>{});
      });

      // process all values in parallel
      std::for_each(pstl::execution::par, jobs_.begin(), jobs_.end(), [&](const auto& job) {
        map[job] = jobs_ | view::transform([&](const auto& j) {
                     return std::pair<Job, common::Distance>{j, distance(j, job)};
                   }) |
          ranges::to_vector | action::sort([](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });

        // map[job] = pairs;
      });
    });
  }

  std::set<Job, compare_jobs> jobs_;
  std::map<std::string, std::map<Job, std::vector<std::pair<Job, common::Distance>>, compare_jobs>> index_;
};
}