#pragma once

#include "algorithms/construction/extensions/Factories.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "utils/Measure.hpp"

#include <chrono>
#include <memory>
#include <set>
#include <vector>

namespace vrp {

/// Provides the way to solve Vehicle Routing Problem.
template<typename AlgorithmDefinition>
class Solver final {
  using Context = algorithms::refinement::RefinementContext;

  /// Represents solution space.
  class SolutionSpace : public ranges::view_facade<SolutionSpace> {
    friend ranges::range_access;

    auto read() const {
      auto start = std::chrono::system_clock::now();

      auto child = refinement_->operator()(*ctx, selector_->operator()(*ctx));
      auto accepted = acceptance_->operator()(*ctx, child);
      terminated_ = termination_->operator()(*ctx, child, accepted);

      if (accepted) {
        ctx->population->push_back(child);
        ranges::action::sort(*ctx->population,
                             [](const auto& lhs, const auto& rhs) { return lhs.second.total() < rhs.second.total(); });
      }

      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);

      return std::tuple(child, duration, accepted);
    }

    bool equal(ranges::default_sentinel_t) const { return terminated_; }

    void next() {
      if (!terminated_) { ++ctx->generation; }
    }

    mutable bool terminated_ = false;
    std::shared_ptr<typename AlgorithmDefinition::Selection> selector_;
    std::shared_ptr<typename AlgorithmDefinition::Refinement> refinement_;
    std::shared_ptr<typename AlgorithmDefinition::Acceptance> acceptance_;
    std::shared_ptr<typename AlgorithmDefinition::Termination> termination_;

  public:
    std::shared_ptr<Context> ctx;

    SolutionSpace() = default;
    explicit SolutionSpace(const std::shared_ptr<const models::Problem>& problem,  //
                           const AlgorithmDefinition& algoDef) :
      ctx(std::make_shared<Context>((*algoDef.template operator()<typename AlgorithmDefinition::Initial>())(problem))),
      selector_(algoDef.template operator()<typename AlgorithmDefinition::Selection>()),
      refinement_(algoDef.template operator()<typename AlgorithmDefinition::Refinement>()),
      acceptance_(algoDef.template operator()<typename AlgorithmDefinition::Acceptance>()),
      termination_(algoDef.template operator()<typename AlgorithmDefinition::Termination>()) {
      ctx->generation = 1;
    }
  };

public:
  models::EstimatedSolution operator()(const std::shared_ptr<const models::Problem>& problem) const {
    auto algoDef = AlgorithmDefinition{problem};
    auto logger = algoDef.template operator()<typename AlgorithmDefinition::Logging>();

    // create solution space within initial solution
    auto space = utils::measure<>::execution_with_result(
      [&]() {
        return SolutionSpace{problem, algoDef};
      },
      [&logger](const auto& result, auto duration) {  //
        logger->operator()(*result.ctx, duration);
      });

    // explore solution space and return best individuum
    return utils::measure<>::execution_with_result(
      [&]() {
        ranges::for_each(space, [&space, &logger](const auto& tuple) {
          const auto& [individuum, duration, accepted] = tuple;
          logger->operator()(*space.ctx, individuum, duration, accepted);
        });
        return space.ctx->population->front();
      },
      [&space, &logger](const auto& result, auto duration) {
        logger->operator()(*space.ctx, result, duration);  //
      });
  }
};
}
