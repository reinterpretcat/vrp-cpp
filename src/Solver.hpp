#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/refinement/acceptance/GreedyAcceptance.hpp"
#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "algorithms/refinement/extensions/RuinAndRecreateSolution.hpp"
#include "algorithms/refinement/extensions/SelectBestSolution.hpp"
#include "algorithms/refinement/logging/LogToNothing.hpp"
#include "algorithms/refinement/termination/MaxIterationCriteria.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "utils/Measure.hpp"

#include <memory>
#include <set>
#include <vector>

namespace vrp {

/// Provides the way to solve Vehicle Routing Problem.
template<typename Initial,      /// Creates initial population.
         typename Selection,    /// Selects individuum from population.
         typename Refinement,   /// Refines individuum.
         typename Acceptance,   /// Accepts individuum.
         typename Termination,  /// Terminates algorithm.
         typename Logging       /// Hook for logging.
         >
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
    mutable Selection selector_;
    mutable Refinement refinement_;
    mutable Acceptance acceptance_;
    mutable Termination termination_;

  public:
    std::shared_ptr<Context> ctx;

    SolutionSpace() = default;
    explicit SolutionSpace(std::shared_ptr<Context> context) :
      ctx(std::move(context)),
      selector_(),
      refinement_(),
      acceptance_(),
      termination_() {
      ctx->generation = 1;
    }
  };

public:
  models::EstimatedSolution operator()(const std::shared_ptr<const models::Problem>& problem) const {
    auto logger = Logging{};

    // create solution space within initial solution
    auto space = utils::measure<>::execution_with_result(
      [&problem]() {
        auto ctx = std::make_shared<Context>(Initial{}(problem));
        return SolutionSpace{ctx};  //
      },
      [&logger](const auto& result, auto duration) {
        logger(*result.ctx, duration);  //
      });

    return space.ctx->population->front();

    //    // explore solution space and return best individuum
    //    return utils::measure<>::execution_with_result(
    //      [&]() {
    //        ranges::for_each(space, [&space, &logger](const auto& pair) {
    //          const auto& [individuum, accepted] = pair;
    //          logger(*space.ctx, individuum, accepted);
    //        });
    //        return space.ctx->population->front();
    //      },
    //      [&space, &logger](const auto& result, auto duration) {
    //        logger(*space.ctx, result, duration);  //
    //      });
  }
};
}
