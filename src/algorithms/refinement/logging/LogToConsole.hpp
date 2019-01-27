#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "streams/out/DumpSolutionAsText.hpp"

#include <chrono>
#include <iostream>

namespace vrp::algorithms::refinement {

/// Logs basic information to console.
struct log_to_console final {
  /// Called when search is started.
  void operator()(const RefinementContext& ctx, std::chrono::milliseconds time) const {
    std::cout << "search for initial population took: " << time.count() << "ms:" << std::endl;
    logIndividuum(ctx.population->front());
  }

  /// Called when new individuum is discovered.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& individuum, bool accepted) const {
    std::cout << (accepted ? "ACCEPTED" : "skipped") << " solution is discovered at generation " << ctx.generation
              << ":" << std::endl;
    logIndividuum(individuum);
  }

  /// Called when search is ended within best solution.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& best, std::chrono::milliseconds time) {
    std::cout << "stopped at generation " << ctx.generation << ", refinement took: " << time.count()
              << "ms, best known individuum is:" << std::endl;
    logIndividuum(ctx.population->front());
  }

private:
  void logIndividuum(const models::EstimatedSolution& es) const {
    // streams::out::dump_solution_as_text{}(std::cout, es);
  }
};
}