#pragma once

#include "models/Problem.hpp"
#include "models/common/Timestamp.hpp"
#include "models/extensions/problem/Distances.hpp"
#include "models/problem/Job.hpp"

#include <map>
#include <pstl/algorithm>
#include <pstl/execution>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

namespace vrp::algorithms::ruin {

/// Calculates job neighborhood in terms of the cost.
struct JobNeighbourhood final {
  explicit JobNeighbourhood(const models::Problem& problem) { initialize(problem); }

  /// Returns range of jobs near to given one.
  ranges::any_view<models::problem::Job> neighbors(const std::string& profile,
                                                   const models::problem::Job& job,
                                                   const models::common::Timestamp time) const {
    return ranges::view::all(index_.find(profile)->second.find(job)->second);
  }

private:
  using Job = models::problem::Job;

  /// Creates job index.
  void initialize(const models::Problem& problem) {
    using namespace ranges;

    ranges::for_each(problem.fleet->profiles(), [&](const auto& profile) {
      auto map = std::map<Job, std::vector<Job>, models::problem::compare_jobs>{};
      auto distance = models::problem::job_distance{};

      std::for_each(pstl::execution::par, problem.jobs.begin(), problem.jobs.end(), [&](const auto& job) {
        auto pairs = problem.jobs | view::transform([&](const auto& j) {
                       return std::pair<double, Job>{distance(j, job), j};
                     }) |
          ranges::to_vector | action::sort([](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        map[job] = pairs | view::transform([](const auto& p) { return p.second; });
      });

      index_[profile] = std::move(map);
    });
  }

  std::map<std::string, std::map<Job, std::vector<Job>, models::problem::compare_jobs>> index_;
};
}