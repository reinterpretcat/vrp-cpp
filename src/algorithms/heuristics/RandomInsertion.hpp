#ifndef VRP_HEURISTICS_RANDOMINSERTION_HPP
#define VRP_HEURISTICS_RANDOMINSERTION_HPP

#include "algorithms/heuristics/Models.hpp"

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

namespace vrp {
namespace algorithms {
namespace heuristics {

/// A helper class which provides the way to get a random customer from solution.
struct find_random_customer {
  EXEC_UNIT find_random_customer(const Context& context, int base);
  EXEC_UNIT vrp::runtime::variant<int, vrp::models::Convolution> operator()();

private:
  const Context& context;
  int base;
  int maxCustomer;
  thrust::uniform_int_distribution<int> dist;
  thrust::minstd_rand rng;
};

/// Implements algorithm of random insertion heuristic.
template<typename TransitionOp>
struct random_insertion final {
  /// Populates individuum with given index starting from task defined by shift.
  EXEC_UNIT void operator()(const Context& context, int index, int shift);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_RANDOMINSERTION_HPP
