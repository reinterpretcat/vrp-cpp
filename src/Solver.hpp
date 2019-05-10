#pragma once

#include "algorithms/construction/extensions/Factories.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "utils/Measure.hpp"

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
      auto child = refinement_(*ctx, selector_(*ctx));
      auto accepted = acceptance_(*ctx, child);
      terminated_ = termination_(*ctx, child, accepted);

      if (accepted) {
        ctx->population->push_back(child);
        ranges::action::sort(*ctx->population,
                             [](const auto& lhs, const auto& rhs) { return lhs.second.total() < rhs.second.total(); });
      }

      return std::pair(child, accepted);
    }

    bool equal(ranges::default_sentinel) const { return terminated_; }

    void next() {
      if (!terminated_) { ++ctx->generation; }
    }

    mutable bool terminated_ = false;
    mutable typename AlgorithmDefinition::Selection selector_;
    mutable typename AlgorithmDefinition::Refinement refinement_;
    mutable typename AlgorithmDefinition::Acceptance acceptance_;
    mutable typename AlgorithmDefinition::Termination termination_;

  public:
    std::shared_ptr<Context> ctx;

    SolutionSpace() = default;
    explicit SolutionSpace(const std::shared_ptr<const models::Problem>& problem,  //
                           const AlgorithmDefinition& algoDef) :
      ctx(std::make_shared<Context>(algoDef.template operator()<typename AlgorithmDefinition::Initial>()(problem))),
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
        logger(*result.ctx, duration);
      });

    // return space.ctx->population->front();

    // explore solution space and return best individuum
    return utils::measure<>::execution_with_result(
      [&]() {
        ranges::for_each(space, [&space, &logger](const auto& pair) {
          const auto& [individuum, accepted] = pair;
          logger(*space.ctx, individuum, accepted);
        });
        return space.ctx->population->front();
      },
      [&space, &logger](const auto& result, auto duration) {
        logger(*space.ctx, result, duration);  //
      });
  }
};
}
