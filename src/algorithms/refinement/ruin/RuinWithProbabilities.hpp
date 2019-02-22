#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms::refinement {

template<std::size_t Nominator, std::size_t Denominator>
struct Probability {
  constexpr static double value = Nominator / static_cast<double>(Denominator);
};

/// Runs several ruin strategies with respect to their probabilities.
template<typename... RuinsWithProbabilities>
struct ruin_with_probabilities final {
  void operator()(const RefinementContext& rCtx,
                  const models::Solution& sln,
                  construction::InsertionContext& iCtx) const {
    auto x = {(runRuin<RuinsWithProbabilities>(rCtx, sln, iCtx), 0)...};
  }

private:
  template<typename RuinWithProbability>
  void runRuin(const RefinementContext& rCtx, const models::Solution& sln, construction::InsertionContext& iCtx) const {
    auto ruinWithProbability = RuinWithProbability{};
    auto probability = std::get<1>(ruinWithProbability).value;

    if (probability >= rCtx.random->uniform<double>(0, 1)) { std::get<0>(ruinWithProbability)(rCtx, sln, iCtx); }
  }
};
}