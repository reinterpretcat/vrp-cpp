#pragma once

#include "LogToSelected.hpp"
#include "algorithms/refinement/acceptance/GreedyAcceptance.hpp"
#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "algorithms/refinement/extensions/RuinAndRecreateSolution.hpp"
#include "algorithms/refinement/extensions/SelectBestSolution.hpp"
#include "algorithms/refinement/termination/VariationCoefficientCriteria.hpp"
#include "models/Problem.hpp"

namespace vrp::example {

/// Creates algorithm definition with predefined parameters.
/// TODO configure solver based on problem?
struct AlgorithmDefinition final {
private:
  template<typename T>
  struct identity {
    typedef T type;
  };

public:
  using Initial = algorithms::refinement::create_refinement_context<>;       /// Creates initial population
  using Selection = algorithms::refinement::select_best_solution;            /// Selects individuum from population.
  using Refinement = algorithms::refinement::ruin_and_recreate_solution<>;   /// Refines individuum.
  using Acceptance = algorithms::refinement::GreedyAcceptance<>;             /// Accepts individuum
  using Termination = algorithms::refinement::VariationCoefficientCriteria;  /// Termination criteria.
  using Logging = log_to_selected;                                           /// Hook for logging.

  explicit AlgorithmDefinition(const std::shared_ptr<const models::Problem>& problem) : problem_(problem) {}

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

  std::shared_ptr<Termination> create(identity<Termination>) const { return std::make_shared<Termination>(100, 0.01); }

  std::shared_ptr<Logging> create(identity<Logging>) const {
    return std::make_shared<Logging>(log_to_selected::LogTo::Console);
  }

  std::shared_ptr<const models::Problem> problem_;
};
}