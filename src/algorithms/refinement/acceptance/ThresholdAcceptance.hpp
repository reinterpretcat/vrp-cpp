#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/common/Cost.hpp"

#include <cmath>

namespace vrp::algorithms::refinement {

/// Threshold-based acceptance defined in "Record Breaking Optimization Results Using
/// the Ruin and Recreate Principle", written by Gerhard Schrimpf, Johannes Schneider,
/// Hermann Stamm-Wilbrandt, and Gunter Dueck.
/// <br>
/// In general, threshold is defined by:
/// T = T0 * exp(-ln 2 * X / alpha)
/// The initial threshold T0 is used to be a half of the standard deviation of the objective
/// function during a random walk, half-life alpha is 0.1, and schedule variable
/// x is increased from 0 to 1 during the optimization run.
struct ThresholdAcceptance final {
  /// Specifies initial threshold.
  common::Cost initial;

  /// Specifies half-life alpha parameter.
  double alpha;

  /// Specifies amount of iterations before acceptance starts to be greedy.
  int iterations;

  bool operator()(const RefinementContext& ctx, const models::Solution& sln) const {
    // TODO
  }

private:
  common::Cost getThreshold(const RefinementContext& ctx) const {
    return ctx.iteration < iterations ? initial * std::exp(-1.0 * std::log(2) * ctx.iteration / (alpha * iterations))
                                      : 0;
  }
};
}
