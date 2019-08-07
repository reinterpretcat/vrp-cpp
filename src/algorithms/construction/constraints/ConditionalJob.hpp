#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "models/extensions/problem/Comparators.hpp"

#include <functional>
#include <mutex>
#include <pstl/algorithm>
#include <pstl/execution>

namespace vrp::algorithms::construction {

/// Allows to assign jobs with some condition.
struct ConditionalJob final : public Constraint {
  /// Specifies a predicate type which returns true when job is considered as required.
  using Predicate = std::function<bool(const InsertionSolutionContext&, const models::problem::Job&)>;

  explicit ConditionalJob(Predicate predicate) : predicate_(std::move(predicate)) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>; }

  /// Accepts solution change.
  void accept(InsertionSolutionContext& ctx) const override {
    // NOTE we scan all jobs as we don't know how much solution has changed.

    analyzeJobs(ctx.required, ctx.ignored, [&](const auto& job) { return !predicate_(ctx, job); });

    analyzeJobs(ctx.ignored, ctx.required, [&](const auto& job) { return predicate_(ctx, job); });
  }

  void accept(InsertionRouteContext&) const override {}

private:
  template<typename F>
  void analyzeJobs(std::vector<models::problem::Job>& src, std::vector<models::problem::Job>& dst, F predicate) const {
    src.erase(std::remove_if(pstl::execution::par,
                             src.begin(),
                             src.end(),
                             [&](const auto& job) {
                               if (predicate(job)) {
                                 std::lock_guard<std::mutex> lock(lock_);
                                 dst.push_back(job);
                                 return true;
                               }
                               return false;
                             }),
              src.end());
  }

  Predicate predicate_;
  mutable std::mutex lock_;
};
}