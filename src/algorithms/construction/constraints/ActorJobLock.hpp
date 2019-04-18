#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "models/JobsLock.hpp"
#include "models/extensions/problem/Comparators.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::construction {

/// Allows to lock specific actors within specific jobs.
struct ActorJobLock final
  : public HardRouteConstraint
  , public HardActivityConstraint {
  constexpr static int Code = 3;

  explicit ActorJobLock(const std::vector<models::JobsLock>& locks, int code = Code) : locks_(), code_(code) {
    ranges::for_each(locks, [&](const auto& l) {
      auto lock = std::make_shared<models::JobsLock>(l);
      ranges::for_each(lock->details, [&](const auto& detail) {
        ranges::for_each(detail.jobs, [&](const auto& j) {
          // TODO check that the same lock is not already there
          locks_[j].push_back(lock);
        });
      });
    });
  }

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(InsertionSolutionContext&) const override {}

  void accept(InsertionRouteContext&) const override {}

  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job& job) const override {
    if (locks_.empty()) return {};

    auto lockPair = locks_.find(job);
    if (lockPair != locks_.end() &&
        ranges::none_of(lockPair->second, [&](const auto& l) { return l->condition(*routeCtx.route->actor); })) {
      return HardRouteConstraint::Result{3};
    }

    return {};
  }

  HardActivityConstraint::Result hard(const InsertionRouteContext& rCtx,
                                      const InsertionActivityContext& aCtx) const override {
    if (locks_.empty()) return {};

    // TODO

    return {};
  }

private:
  int code_;
  std::map<models::problem::Job,                            //
           std::vector<std::shared_ptr<models::JobsLock>>,  //
           models::problem::compare_jobs>
    locks_;
};
}