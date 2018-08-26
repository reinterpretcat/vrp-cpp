#include "algorithms/heuristics/CheapestInsertion.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"


using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Alias for customer variant.
using Customer = variant<int, Convolution>;

}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

EXEC_UNIT find_random_customer::find_random_customer(const Context& context, int base) :
  context(context), base(base), maxCustomer(context.tasks.customers - 1), dist(1, maxCustomer),
  rng() {}


EXEC_UNIT Customer find_random_customer::operator()() {
  auto start = dist(rng);
  auto customer = start;
  bool increment = start % 2 == 0;

  do {
    Plan plan = context.tasks.plan[base + customer];

    if (!plan.isAssigned()) {
      return plan.hasConvolution()
               ? Customer::create<Convolution>(*(context.convolutions + plan.convolution()))
               : Customer::create<int>(customer);
    }

    // try to find next customer
    if (increment)
      customer = customer == maxCustomer ? 1 : customer + 1;
    else
      customer = customer == 1 ? maxCustomer : customer - 1;
  } while (customer != start);

  return Customer::create<int>(-1);
}


template<typename TransitionOp>
EXEC_UNIT void random_insertion<TransitionOp>::operator()(const Context& context,
                                                          int index,
                                                          int shift) {
  cheapest_insertion<TransitionOp, find_random_customer>{}(context, index, shift);
}

/// NOTE make linker happy.
template class random_insertion<TransitionOperator>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
