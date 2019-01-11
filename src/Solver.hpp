#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/refinement/acceptance/ThresholdAcceptance.hpp"
#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "algorithms/refinement/extensions/LogToConsole.hpp"
#include "algorithms/refinement/extensions/SelectBestSolution.hpp"
#include "algorithms/refinement/rar/RuinAndRecreateSolution.hpp"
#include "algorithms/refinement/termination/MaxIterationCriteria.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

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
      auto child = refinement_(ctx_, selector_(ctx_));
      auto accepted = acceptance_(ctx_, child);
      terminated_ = termination_(ctx_, child, accepted);
      if (accepted) {
        // ctx_.population->push_back(child);
        // TODO sort population by cost
      }

      return std::pair(child, accepted);
    }

    bool equal(ranges::default_sentinel) const { return terminated_; }

    void next() { ++ctx_.generation; }

    Context ctx_;

    mutable bool terminated_ = false;
    mutable Selection selector_;
    mutable Refinement refinement_;
    mutable Acceptance acceptance_;
    mutable Termination termination_;

  public:
    SolutionSpace() = default;
    explicit SolutionSpace(const Context& ctx) : ctx_(ctx), selector_(), refinement_(), acceptance_(), termination_(){};
  };

public:
  models::EstimatedSolution operator()(const models::Problem& problem) const {
    auto logger = Logging{};
    auto ctx = Initial{}(problem);
    auto space = SolutionSpace{ctx};

    logger(ctx);

    auto last = ranges::accumulate(space, 0, [&](int generation, const auto& pair) {
      const auto& [individuum, accepted] = pair;
      ctx.generation = generation;
      logger(ctx, individuum, accepted);
      return generation + 1;
    });

    logger(ctx, last);

    return ctx.population->front();
  }
};

using DefaultSolver = Solver<algorithms::refinement::create_refinement_context<>,
                             algorithms::refinement::select_best_solution,
                             algorithms::refinement::ruin_and_recreate_solution<>,
                             algorithms::refinement::ThresholdAcceptance<>,
                             algorithms::refinement::MaxIterationCriteria,
                             algorithms::refinement::log_to_console>;
}