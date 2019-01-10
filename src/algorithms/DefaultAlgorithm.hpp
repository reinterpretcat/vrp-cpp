#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/heuristics/CheapestInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms {

/// Specifies default algorithm logic.
struct DefaultAlgorithm final {
  /// Specifies population type which holds information about population including generation.
  using Population = refinement::RefinementContext;
  /// Specifies single  individuum type.
  using Individuum = Population::Individuum;

  /// Creates initial population for the problem.
  struct create_population final {
    Population operator()(const models::Problem& problem) const {
      // TODO
    }
  };

  /// Selects individuum from population.
  struct select_individuum final {
    Individuum operator()(const Population& population) const {
      // TODO
    }
  };

  /// Refines individuum.
  struct refine_individuum final {
    Individuum operator()(const Population& population, const Individuum& individuum) const {
      // TODO
    }
  };

  /// Accepts individuum.
  struct accept_individuum final {
    bool operator()(const Population& population, const Individuum& individuum) const {
      // TODO
    }
  };

  /// Terminates individuum.
  struct terminate_algorithm final {
    bool operator()(const Population& population, const Individuum& individuum, bool accepted) {
      // TODO
    }
  };

  /// Logs progress.
  struct log_progress final {
    /// Called after initial population is created.
    void operator()(const Population& population) const {}
    /// Called after iteration is completed.
    void operator()(const Population& population, const Individuum& individuum) const {}
    /// Called after algorithm is terminated.
    void operator()(int iteration) const {}
  };

  using Initial = create_population;        /// Creates initial population.
  using Selection = select_individuum;      /// Selects individuum from population for given iteration.
  using Refinement = refine_individuum;     /// Refines individuum.
  using Acceptance = accept_individuum;     /// Accepts individuum.
  using Termination = terminate_algorithm;  /// Terminates algorithm.
  using Logging = log_progress;             /// Hook for logging
};
}