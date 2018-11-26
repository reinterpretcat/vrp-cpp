#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/common/Cost.hpp"

#include <memory>
#include <rxcpp/rx.hpp>

namespace vrp::algorithms::construction {

/// Cheapest insertion
struct CheapestInsertion final : InsertionHeuristic<CheapestInsertion> {
  explicit CheapestInsertion(const std::shared_ptr<const InsertionEvaluator>& evaluator) : evaluator_(evaluator) {}

  InsertionContext analyze(const InsertionContext& ctx) const {
    // TODO
    rxcpp::observable<>::iterate(ctx.jobs).map([&](const auto& job) {
      auto result = evaluator_->evaluate(job, ctx);
      return job;
    });
    //.reduce();

    return {};
  }

private:
  std::shared_ptr<const InsertionEvaluator> evaluator_;
};
}