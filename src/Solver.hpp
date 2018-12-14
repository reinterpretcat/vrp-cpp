#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

#include <memory>
#include <set>
#include <vector>

namespace vrp {

/// Provides the way to solve Vehicle Routing Problem.
template<typename Initial,     /// Creates initial population.
         typename Selection,   /// Selects individuum from population for given iteration.
         typename Refinement,  /// Refines individuum.
         typename Acceptance,  /// Accepts individuum.
         typename Termination  /// Terminates search.
         >
class Solver final {
  /// Represents solution space.
  class SolutionSpace : public ranges::view_facade<SolutionSpace> {
    friend ranges::range_access;

    const auto& read() const { return selector_(iteration_); }

    bool equal(ranges::default_sentinel) const { return termination_(iteration_); }

    void next() { ++iteration_; }

    int iteration_ = 0;

    Selection selector_;
    Termination termination_;

  public:
    explicit SolutionSpace(Selection selector, Termination termination) :
      selector_(std::move(selector)),
      termination_(std::move(termination)){};
  };

public:
  std::string operator()(const std::string& problem) const {
    auto population = Initial{}(problem);
    auto refinement = Refinement{population};
    auto acceptance = Acceptance{population};

    ranges::for_each(SolutionSpace{Selection{population}, Termination{population}},
                     [&](const auto& individuum) { acceptance(refinement(individuum)); });

    return population->front();
  }
};
}