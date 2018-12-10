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
  Jobs(Jobs&& other) : jobs_(std::move(other.jobs_)), index_(std::move(other.index_)) {}
  Jobs(const Jobs&) = delete;
  Jobs& operator=(const Jobs&) = delete;

  /// Returns all jobs.
  ranges::any_view<Job> all() const { return ranges::view::all(jobs_); }

  /// Returns range of jobs near to given one.
  ranges::any_view<Job> neighbors(const std::string& profile, const Job& job, const common::Timestamp time) const {
    return ranges::view::all(index_.find(profile)->second.find(job)->second);
  }

private:
  void createJobSet(ranges::any_view<Job>& jobs) { ranges::copy(jobs, ranges::inserter(jobs_, jobs_.begin())); }

  /// Creates job index.
  void createJobIndex(const costs::TransportCosts& transport, ranges::any_view<std::string>& profiles) {
    using namespace ranges;

    ranges::for_each(profiles, [&](const auto& profile) {
      auto map = std::map<Job, std::vector<Job>, compare_jobs>{};
      auto distance = models::problem::job_distance{};

      std::for_each(pstl::execution::par, jobs_.begin(), jobs_.end(), [&](const auto& job) {
        auto pairs = jobs_ | view::transform([&](const auto& j) {
                       return std::pair<double, Job>{distance(j, job), j};
                     }) |
          ranges::to_vector | action::sort([](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        map[job] = pairs | view::transform([](const auto& p) { return p.second; });
      });

      index_[profile] = std::move(map);
    });
  }

  std::set<Job, compare_jobs> jobs_;
  std::map<std::string, std::map<Job, std::vector<Job>, compare_jobs>> index_;
};
}