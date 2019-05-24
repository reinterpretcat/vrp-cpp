#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "algorithms/refinement/logging/LogToExtras.hpp"
#include "algorithms/refinement/logging/LogToNothing.hpp"
#include "models/Solution.hpp"

#include <chrono>

namespace vrp::example {

/// Logs information to specific logger.
struct log_to_selected final {
  /// Specifies types of supported logging.
  class LogTo {
  public:
    /// Logs to nothing (NOP).
    constexpr static int Nothing = 1 << 0;
    /// Logs to console.
    constexpr static int Console = 1 << 1;
    /// Logs to extras.
    constexpr static int Extras = 1 << 2;
  };

  explicit log_to_selected(int logTo) : logTo_(logTo) {}

  /// Called when search is started.
  void operator()(const algorithms::refinement::RefinementContext& ctx, std::chrono::milliseconds time) {
    if (logTo_ & LogTo::Console) logToConsole_(ctx, time);

    if (logTo_ & LogTo::Extras) logToExtras_(ctx, time);

    if (logTo_ & LogTo::Nothing) logToNothing_(ctx, time);
  }

  /// Called when new individuum is discovered.
  void operator()(const algorithms::refinement::RefinementContext& ctx,
                  const models::EstimatedSolution& individuum,
                  std::chrono::milliseconds duration,
                  bool accepted) {
    if (logTo_ & LogTo::Console) logToConsole_(ctx, individuum, duration, accepted);

    if (logTo_ & LogTo::Extras) logToExtras_(ctx, individuum, duration, accepted);

    if (logTo_ & LogTo::Nothing) logToNothing_(ctx, individuum, duration, accepted);
  }

  /// Called when search is ended within best solution.
  void operator()(const algorithms::refinement::RefinementContext& ctx,
                  const models::EstimatedSolution& best,
                  std::chrono::milliseconds duration) {
    if (logTo_ & LogTo::Console) logToConsole_(ctx, best, duration);

    if (logTo_ & LogTo::Extras) logToExtras_(ctx, best, duration);

    if (logTo_ & LogTo::Nothing) logToNothing_(ctx, best, duration);
  }

private:
  int logTo_;

  algorithms::refinement::log_to_console logToConsole_;
  algorithms::refinement::log_to_extras logToExtras_;
  algorithms::refinement::log_to_nothing logToNothing_;
};
}