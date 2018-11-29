#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/common/Cost.hpp"

#include <memory>
#include <rxcpp/rx.hpp>

namespace vrp::algorithms::construction {

/// Cheapest insertion heuristic.
struct CheapestInsertion final : InsertionHeuristic<CheapestInsertion> {
  CheapestInsertion(const std::shared_ptr<const InsertionEvaluator>& evaluator) : evaluator_(evaluator) {}

  InsertionContext analyze(const InsertionContext& ctx) {
    auto newCtx = InsertionContext(ctx);
    while (!newCtx.jobs.empty()) {
      // TODO use C++17 parallel algorithms instead of rxcpp once it has better runtime support
      rxcpp::observable<>::iterate(newCtx.jobs)
        .map([&](const auto& job) { return evaluator_->evaluate(job, newCtx); })
        .reduce(make_result_failure(),
                [](const auto& acc, const auto& result) { return get_cheapest(acc, result); },
                [](const auto& res) { return res; })
        .as_blocking()
        .last()
        .visit(ranges::overload(
          [&](const InsertionSuccess& success) {
            // perform insertion
            success.route.first->actor = success.actor;
            ctx.registry->use(*success.actor);

            ranges::for_each(success.activities, [&](const auto& act) {

            });
            newCtx.jobs.erase(success.job);
          },
          [&](const InsertionFailure& failure) {
            //            ranges::push_back(newCtx.unassigned, newCtx.jobs | ranges::view::transform([&](const auto&
            //            job) {
            //                                                   return std::pair(job, failure.constraint);
            //                                                 }));
            newCtx.jobs.clear();
          }));
    }
    return newCtx;
  }

private:
  std::shared_ptr<const InsertionEvaluator> evaluator_;
};
}