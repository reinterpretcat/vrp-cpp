#pragma once

#include "Solver.hpp"
#include "algorithms/refinement/acceptance/GreedyAcceptance.hpp"
#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "algorithms/refinement/extensions/RuinAndRecreateSolution.hpp"
#include "algorithms/refinement/extensions/SelectBestSolution.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "algorithms/refinement/termination/MaxIterationCriteria.hpp"

namespace vrp::test {

/// Creates solver with default algorithms.
template<typename LoggingType = vrp::algorithms::refinement::log_to_console>
struct create_default_solver final {
  /// Creates algorithm definition with predefined parameters.
  struct DefaultAlgorithmDefinition final {
  private:
    template<typename T>
    struct identity {
      typedef T type;
    };

  public:
    using Initial = algorithms::refinement::create_refinement_context<>;      /// Creates initial population
    using Selection = algorithms::refinement::select_best_solution;           /// Selects individuum from population.
    using Refinement = algorithms::refinement::ruin_and_recreate_solution<>;  /// Refines individuum.
    using Acceptance = algorithms::refinement::GreedyAcceptance<>;            /// Accepts individuum
    using Termination = algorithms::refinement::MaxIterationCriteria;         /// Terminates algorithm.
    using Logging = LoggingType;                                              /// Hook for logging.

    explicit DefaultAlgorithmDefinition(const std::shared_ptr<const models::Problem>& problem) : problem_(problem) {}

    template<typename T>
    std::shared_ptr<T> operator()() const {
      return create(identity<T>());
    }

  private:
    template<typename T>
    std::shared_ptr<T> create(identity<T>) const {
      throw std::domain_error("Unknown algorithm");
    }

    std::shared_ptr<Initial> create(identity<Initial>) const { return std::make_shared<Initial>(); }

    std::shared_ptr<Selection> create(identity<Selection>) const { return std::make_shared<Selection>(); }

    std::shared_ptr<Refinement> create(identity<Refinement>) const { return std::make_shared<Refinement>(); }

    std::shared_ptr<Acceptance> create(identity<Acceptance>) const { return std::make_shared<Acceptance>(); }

    std::shared_ptr<Termination> create(identity<Termination>) const { return std::make_shared<Termination>(); }

    std::shared_ptr<Logging> create(identity<Logging>) const { return std::make_shared<Logging>(); }


    std::shared_ptr<const models::Problem> problem_;
  };

  auto operator()() const { return Solver<DefaultAlgorithmDefinition>{}; }
};

/// Solves problem from stream using default solver.
template<typename Stream, typename Reader>
struct solve_stream final {
  std::pair<std::shared_ptr<const models::Problem>, std::shared_ptr<const models::Solution>> operator()() const {
    auto stream = Stream{}();
    auto problem = Reader{}.operator()(stream);

    auto solver = create_default_solver{}();
    auto estimatedSolution = solver(problem);

    return {problem, estimatedSolution.first};
  }
};
}