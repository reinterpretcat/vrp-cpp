#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

#include <memory>
#include <set>
#include <vector>

namespace vrp {

/// Provides the way to solve Vehicle Routing Problem.
template<typename Algorithm>
class Solver final {
  using Initial = typename Algorithm::Initial;
  using Selection = typename Algorithm::Selection;
  using Refinement = typename Algorithm::Refinement;
  using Acceptance = typename Algorithm::Acceptance;
  using Termination = typename Algorithm::Termination;
  using Logging = typename Algorithm::Logging;

  /// Represents solution space.
  class SolutionSpace : public ranges::view_facade<SolutionSpace> {
    friend ranges::range_access;

    const auto& read() const {
      auto child = refinement_(selector_(iteration_), iteration_);
      auto accepted = acceptance_(child, iteration_);
      terminated_ = termination_(child, iteration_, accepted);
      return child;
    }

    bool equal(ranges::default_sentinel) const { return terminated_; }

    void next() { ++iteration_; }

    int iteration_ = 0;
    bool terminated_ = false;

    Selection selector_;
    Refinement refinement_;
    Acceptance acceptance_;
    Termination termination_;

  public:
    SolutionSpace(Selection selector, Refinement refinement, Acceptance acceptance, Termination termination) :
      selector_(std::move(selector)),
      refinement_(std::move(refinement)),
      acceptance_(std::move(acceptance)),
      termination_(std::move(termination)){};
  };

public:
  models::Solution operator()(const models::Problem& problem) const {
    auto logger = Logging{};
    auto population = Initial{}(problem);

    logger(population);

    auto last = ranges::accumulate(
      SolutionSpace{Selection{population}, Refinement{population}, Acceptance{population}, Termination{population}},
      0,
      [&](const int iteration, const auto& individuum) {
        logger(individuum, iteration);
        return iteration + 1;
      });

    logger(last);

    return population.best();
  }
};
}