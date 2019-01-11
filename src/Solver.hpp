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

    const auto& read() const {
      auto child = refinement_(ctx_, selector_(ctx_));
      auto accepted = acceptance_(ctx_, child);
      terminated_ = termination_(ctx_, child, accepted);
      if (accepted) {
        ctx_.population->push_back(child);
        // TODO sort population by cost
      }

      return child;
    }

    bool equal(ranges::default_sentinel) const { return terminated_; }

    void next() { ++ctx_.generation; }

    bool terminated_ = false;

    Context& ctx_;
    Selection selector_;
    Refinement refinement_;
    Acceptance acceptance_;
    Termination termination_;

  public:
    SolutionSpace(Context& ctx,
                  Selection selector,
                  Refinement refinement,
                  Acceptance acceptance,
                  Termination termination) :
      ctx_(ctx),
      selector_(std::move(selector)),
      refinement_(std::move(refinement)),
      acceptance_(std::move(acceptance)),
      termination_(std::move(termination)){};
  };

public:
  models::Solution operator()(const models::Problem& problem) const {
    auto logger = Logging{};
    auto ctx = Initial{}(problem);
    auto space = SolutionSpace{ctx, Selection{}, Refinement{}, Acceptance{}, Termination{}};

    logger(ctx);

    auto last = ranges::accumulate(space, 0, [&](const int generation, const auto& individuum) {
      logger(individuum, generation);
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