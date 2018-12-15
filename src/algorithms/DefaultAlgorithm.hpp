#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/heuristics/CheapestInsertion.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms {

/// Specifies default algorithm logic.
template<typename Heuristic = construction::CheapestInsertion<construction::InsertionEvaluator>>
struct DefaultAlgorithm final {
  using Individuum = construction::InsertionContext;
  using Population = std::shared_ptr<std::vector<Individuum>>;

  /// Creates initial population.
  struct create_population final {
    Population operator()(const models::Problem& problem) const {}

    models::Solution best() const { return {}; }
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
    void operator()(const Individuum& individuum, int iteration) const {}
  };

  using Initial = create_population;        /// Creates initial population.
  using Selection = select_individuum;      /// Selects individuum from population for given iteration.
  using Refinement = refine_individuum;     /// Refines individuum.
  using Acceptance = accept_individuum;     /// Accepts individuum.
  using Termination = terminate_algorithm;  /// Terminates algorithm.
  using Logging = log_progress;             /// Hook for logging
};
}