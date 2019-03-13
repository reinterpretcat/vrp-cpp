#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "models/extensions/problem/Comparators.hpp"

namespace vrp::algorithms::construction {

/// Allows to lock specific actors within specific jobs.
struct ActorJobLock final : public HardRouteConstraint {
  explicit ActorJobLock(int code = 3) : code_(code), locks_() {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(InsertionSolutionContext&) const override {}

  void accept(InsertionRouteContext&) const override {}

  /// Locks actor within job.
  ActorJobLock& lock(const std::shared_ptr<const models::solution::Actor>& actor, const models::problem::Job& job) {
    locks_[job].insert(actor);

    return *this;
  }

  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job& job) const override {
    auto jobLocks = locks_.find(job);

    if (jobLocks != locks_.end() && jobLocks->second.find(routeCtx.route->actor) == jobLocks->second.end())
      return HardRouteConstraint::Result{3};

    return {};
  }

private:
  int code_;
  std::map<models::problem::Job,                                      //
           std::set<std::shared_ptr<const models::solution::Actor>>,  //
           models::problem::compare_jobs>
    locks_;
};
}