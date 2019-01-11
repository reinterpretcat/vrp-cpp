#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"

#include <iostream>

namespace vrp::algorithms::refinement {

/// Logs basic information to console.
struct log_to_console final {
  /// Called when context is created.
  void operator()(const RefinementContext& ctx) const {
    std::cout << "initial solution:\n";
    logIndividuum(ctx.population->front());
  }

  /// Called when new individuum is discovered.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& individuum, bool accepted) const {
    std::cout << "new " << (accepted ? "accepted" : "skipped") << " solution is discovered at generation "
              << ctx.generation << ":" << std::endl;
    logIndividuum(individuum);
  }

  /// Called when search is completed
  void operator()(const RefinementContext& ctx, int generation) const {
    std::cout << "stopped at generation " << generation << ", best known is:" << std::endl;
    logIndividuum(ctx.population->front());
  }

private:
  void logIndividuum(const models::EstimatedSolution& es) const {
    std::cout << "\t\tcost:" << es.second.actual << " + " << es.second.penalty
              << "\n\t\troutes:" << es.first->routes.size() << std::endl;
  }
};

/// Dummy logging which does nothing.
struct log_to_nothing final {
  /// Called when context is created.
  void operator()(const RefinementContext& ctx) const {}

  /// Called when new individuum is discovered.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& individuum, bool accepted) const {}

  /// Called when search is completed
  void operator()(const RefinementContext& ctx, int generation) const {}
};
}