#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "streams/out/DumpSolutionAsText.hpp"

#include <chrono>
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

  /// Called when search is completed.
  void operator()(const RefinementContext& ctx, std::chrono::milliseconds::rep time) const {
    std::cout << "stopped at generation " << ctx.generation << ", took: " << time << "ms, best known is:" << std::endl;
    logIndividuum(ctx.population->front());
  }

private:
  void logIndividuum(const models::EstimatedSolution& es) const {
    streams::out::dump_solution_as_text{}(std::cout, es);
  }
};
}