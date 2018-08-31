#ifndef VRP_ALGORITHMS_GENETIC_MUTATIONS_HPP
#define VRP_ALGORITHMS_GENETIC_MUTATIONS_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Creates mutant from given individuum using best convolutions.
template<typename TransitionOp>
struct mutate_weak_subtours final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;

  EXEC_UNIT void operator()(const Mutation& mutation) const;
};

/// Creates mutant from given individuum destroying specific tours.
template<typename TransitionOp>
struct mutate_weak_tours final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;

  EXEC_UNIT void operator()(const Mutation& mutation) const;
};

/// Do nothing.
struct empty_mutator final {
  ANY_EXEC_UNIT void operator()(const Mutation& mutation) const {}
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_MUTATIONS_HPP
