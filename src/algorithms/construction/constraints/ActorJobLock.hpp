#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "models/Lock.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/solution/Helpers.hpp"
#include "models/solution/Actor.hpp"

#include <map>
#include <range/v3/all.hpp>
#include <unordered_map>
#include <unordered_set>

namespace vrp::algorithms::construction {

/// Allows to lock specific actors within specific jobs using different rules.
struct ActorJobLock final
  : public HardRouteConstraint
  , public HardActivityConstraint {
  constexpr static int Code = 3;

private:
  /// Represents a simple job index.
  struct JobIndex final {
    models::problem::Job first;
    models::problem::Job last;
    std::unordered_set<models::problem::Job,  //
                       models::problem::hash_job,
                       models::problem::is_the_same_jobs>
      jobs;

    bool contains(const models::problem::Job& job) const { return jobs.find(job) != jobs.end(); }
  };

  /// Represents a rule created from lock model.
  struct Rule final {
    /// Actor condition.
    std::shared_ptr<models::Lock::Condition> condition;
    /// Has departure in the beginning.
    bool hasDep;
    /// Has arrival in the end.
    bool hasArr;
    /// Stores jobs.
    JobIndex index;
  };

public:
  explicit ActorJobLock(const std::vector<models::Lock>& locks, int code = Code) :
    code_(code),
    conditions_(),
    rules_(),
    initRules_() {
    ranges::for_each(locks, [&](const auto& lock) {
      auto condition = std::make_shared<models::Lock::Condition>(lock.condition);

      ranges::for_each(lock.details, [&](const auto& detail) {
        // NOTE create rule only for strict order
        if (detail.order == models::Lock::Order::Strict) {
          assert(!detail.jobs.empty());

          auto rule = std::make_shared<Rule>();

          rule->condition = condition;
          rule->hasDep = detail.position.stickToDeparture;
          rule->hasArr = detail.position.stickToArrival;
          rule->index.first = detail.jobs.front();
          rule->index.last = detail.jobs.back();
          ranges::copy(detail.jobs, ranges::inserter(rule->index.jobs, rule->index.jobs.begin()));

          initRules_.push_back(rule);
        }

        ranges::for_each(detail.jobs, [&](const auto& j) {
          // NOTE check in O(N) that the same condition is not already in the collection.
          if (ranges::none_of(conditions_[j], [&](const auto& exl) { return exl == condition; })) {
            conditions_[j].push_back(condition);
          }
        });
      });
    });
  }

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(InsertionSolutionContext& ctx) const override {
    // NOTE initialize rules collection once. We do it here as constraint's constructor
    // cannot depend on models from solution domain (Registry in this case).
    if (!initRules_.empty()) {
      ranges::for_each(initRules_, [&](const auto& rule) {
        ranges::for_each(
          ctx.registry->all() | ranges::view::filter([&](const auto& a) { return (*rule->condition)(*a); }),
          [&](const auto& actor) {
            // TODO assert arrival position is not set in rule for open VRP
            rules_[actor].push_back(rule);
          });
      });

      /// Mark actor as used
      ranges::for_each(rules_, [&](const auto& pair) { ctx.registry->use(pair.first); });

      initRules_.clear();
    }
  }

  void accept(InsertionRouteContext&) const override {}

  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job& job) const override {
    auto condPair = conditions_.find(job);
    return condPair != conditions_.end() &&
        ranges::none_of(condPair->second, [&](const auto& cond) { return (*cond)(*routeCtx.route->actor); })
      ? HardRouteConstraint::Result{3}
      : HardRouteConstraint::Result{};
  }

  HardActivityConstraint::Result hard(const InsertionRouteContext& rCtx,
                                      const InsertionActivityContext& aCtx) const override {
    using namespace vrp::models::problem;
    using namespace vrp::models::solution;

    auto rulePair = rules_.find(rCtx.route->actor);
    return rulePair != rules_.end() &&
        ranges::any_of(  //
             rulePair->second,
             [&](const auto& rule) {
               // check with departure
               bool hasPrev = false;
               auto prev = retrieve_job{}(*aCtx.prev);
               if (prev) {
                 hasPrev = rule->index.contains(prev.value());
                 if (rule->hasDep && hasPrev && !is_the_same_jobs{}(prev.value(), rule->index.last)) return true;
               } else if (rule->hasDep)
                 return true;

               // check with arrival
               bool hasNext = false;
               if (aCtx.next) {
                 auto next = retrieve_job{}(*aCtx.next.value());
                 if (next) {
                   hasNext = rule->index.contains(next.value());
                   if (rule->hasArr && hasNext && !is_the_same_jobs{}(next.value(), rule->index.first)) return true;
                 } else if (rule->hasArr)
                   return true;
               }

               // check general
               return hasPrev && hasNext;
             })
      ? stop(code_)
      : success();
  }

private:
  int code_;
  std::map<models::problem::Job,                                   //
           std::vector<std::shared_ptr<models::Lock::Condition>>,  //
           models::problem::compare_jobs>
    conditions_;

  mutable std::unordered_map<std::shared_ptr<const models::solution::Actor>,  //
                             std::vector<std::shared_ptr<Rule>>>
    rules_;

  mutable std::vector<std::shared_ptr<Rule>> initRules_;
};
}