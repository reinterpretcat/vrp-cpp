#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/refinement/acceptance/GreedyAcceptance.hpp"
#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "algorithms/refinement/extensions/SelectBestSolution.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "algorithms/refinement/rar/RuinAndRecreateSolution.hpp"
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
      auto child = refinement_(ctx, selector_(ctx));
      auto accepted = acceptance_(ctx, child);
      terminated_ = termination_(ctx, child, accepted);

      if (accepted) {
        ctx.population->push_back(child);
        ranges::action::sort(*ctx.population,
                             [](const auto& lhs, const auto& rhs) { return lhs.second.total() < rhs.second.total(); });
      }

      return std::pair(child, accepted);
    }

    bool equal(ranges::default_sentinel) const { return terminated_; }

    void next() { ++ctx.generation; }

    mutable bool terminated_ = false;
    mutable Selection selector_;
    mutable Refinement refinement_;
    mutable Acceptance acceptance_;
    mutable Termination termination_;

  public:
    Context ctx;

    SolutionSpace() = default;
    explicit SolutionSpace(const models::Problem& problem) :
      ctx(Initial{}(problem)),
      selector_(),
      refinement_(),
      acceptance_(),
      termination_(){};
  };

public:
  models::EstimatedSolution operator()(const models::Problem& problem) const {
    auto logger = Logging{};
    auto space = SolutionSpace{problem};

    logger(space.ctx);

    auto time = utils::measure<>::execution([&]() {
      ranges::accumulate(space, 1, [&](int generation, const auto& pair) {
        const auto& [individuum, accepted] = pair;
        space.ctx.generation = generation;
        logger(space.ctx, individuum, accepted);
        return generation + 1;
      });
    });

    logger(space.ctx, time);

    return space.ctx.population->front();
  }
};

using DefaultSolver = Solver<algorithms::refinement::create_refinement_context<>,
                             algorithms::refinement::select_best_solution,
                             algorithms::refinement::ruin_and_recreate_solution<>,
                             algorithms::refinement::GreedyAcceptance<>,
                             algorithms::refinement::MaxIterationCriteria,
                             algorithms::refinement::log_to_console>;
}