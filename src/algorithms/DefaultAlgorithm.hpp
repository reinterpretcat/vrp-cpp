#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/heuristics/CheapestInsertion.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms {

/// Specifies default algorithm logic.
template<typename Heuristic = construction::CheapestInsertion>
struct DefaultAlgorithm final {
  using Individuum = construction::InsertionContext;

  /// Represents population entity.
  struct Population final {
    std::shared_ptr<std::vector<Individuum>> individuums = std::make_shared<std::vector<Individuum>>();

    /// Returns best individuum as solution.
    models::Solution best() {
      // TODO
    }
  };

  /// Creates initial population.
  struct create_population final {
    Population operator()(const models::Problem& problem) const {}
  };

  /// Selects individuum from population.
  struct select_individuum final {
    Population population;

    const Individuum& operator()(int iteration) const { return {}; }
  };

  /// Refines individuum.
  struct refine_individuum final {
    Population population;

    Individuum operator()(const Individuum& parent, int iteration) const { return {}; }
  };

  /// Accepts individuum.
  struct accept_individuum final {
    Population population;

    bool operator()(const Individuum& individuum, int iteration) const {}
  };

  /// Terminates individuum.
  struct terminate_algorithm final {
    bool operator()(const Individuum& individuum, int iteration, bool accepted) {}
  };

  /// Logs progress.
  struct log_progress final {
    /// Called after initial population is created.
    void operator()(const Population& population) const {}
    /// Called after iteration is completed.
    void operator()(const Individuum& individuum, int iteration) const {}
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