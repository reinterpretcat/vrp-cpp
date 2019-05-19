#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/common/Cost.hpp"

#include <map>
#include <vector>

namespace vrp::algorithms::refinement {

/// Logs iteration information .
struct log_to_extras final {
  inline const static std::string ExtrasKey = "iterations";

  /// Defines iteration information.
  struct Iteration final {
    /// Iteration numbser
    size_t number;
    /// Best known cost.
    models::common::ObjectiveCost cost;
    /// Amount of routes.
    size_t routes;
    /// Amount of unassigned jobs.
    size_t unassigned;
    /// Iteration duration in milliseconds.
    long duration;
    /// Overall duration in milliseconds since start.
    long timestamp;
  };

  /// Called when search is started.
  void operator()(const RefinementContext& ctx, std::chrono::milliseconds duration) {
    time_ = duration.count();
    logIteration(ctx.population->front(), duration);
  }

  /// Called when new individuum is discovered.
  void operator()(const RefinementContext& ctx,
                  const models::EstimatedSolution& individuum,
                  std::chrono::milliseconds duration,
                  bool accepted) {
    logIteration(accepted ? individuum : last_, duration);
  }

  /// Called when search is ended within best solution.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& best, std::chrono::milliseconds time) {
    // NOTE modifying const data
    auto& solution = const_cast<models::Solution&>(*best.first);

    if (!best.first->extras) { solution.extras = std::make_shared<std::map<std::string, std::any>>(); }

    const_cast<std::map<std::string, std::any>&>(*solution.extras)[ExtrasKey] = iterations_;
  }

private:
  void logIteration(const models::EstimatedSolution& individuum, std::chrono::milliseconds duration) {
    time_ += duration.count();

    iterations_.push_back(Iteration{
      ++number_,
      individuum.second,
      individuum.first->routes.size(),
      individuum.first->unassigned.size(),
      duration.count(),
      time_,
    });

    last_ = individuum;
  }

  size_t number_ = 0;
  long time_ = 0;
  models::EstimatedSolution last_ = {};
  std::vector<Iteration> iterations_ = {};
};
}