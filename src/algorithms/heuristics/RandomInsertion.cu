#include "algorithms/heuristics/Operators.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

namespace {
/// Finds next random customer to serve.
struct find_random_customer final {
  Tasks::Shadow tasks;
  int operator()(int seed) {
    // TODO
    return 0;
  }
};

/// Finds the "best" transition for given customer
struct find_best_arc final {
  const int begin;

  /// @param from      Task to start from.
  /// @param to        Next task.
  /// @param customer  Customer id.
  /// @param vehicle   Vehicle id.
  /// @param index     First task where vehicle is used.
  Transition operator()(int from, int to, int customer, int vehicle, int index) {
    // TODO

    return Transition();
  }
};
}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

template<typename TransitionOp>
void random_insertion<TransitionOp>::operator()(const Context& context, int index, int shift) {
  const auto begin = index * context.problem.size;

  auto transitionOp = TransitionOp{context.problem, context.tasks};
  auto customerOp = find_random_customer{context.tasks};
  auto arcOp = find_best_arc{begin};

  int seed = 0;

  int from = shift;
  int to = from + 1;
  int vehicle = context.tasks.vehicles[from];

  // tracks which task start new vehicle, shift is slightly ignored for efficiency
  int vStart = from;

  do {
    auto customer = customerOp(seed);
    auto transition = arcOp(from, to, customer, vehicle, vStart);
    if (transition.isValid()) {
      auto cost = transitionOp.estimate(transition);
      from = transitionOp.perform(transition, cost);
      to = from + 1;
    } else {
      // NOTE cannot find any further customer to serve within vehicle
      if (from == 0 || vehicle == context.problem.resources.vehicles - 1) break;

      from = 0;
      spawn_vehicle(context.problem, context.tasks, from, ++vehicle);
    }
  } while (to < context.problem.size);
}

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
