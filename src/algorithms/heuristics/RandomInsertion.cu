#include "algorithms/heuristics/Operators.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"

#include <iterators/Aggregates.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {
/// Stores insertion point and its cost together.
using InsertionCost = thrust::pair<int, float>;

/// Specifies vehicle range: start, end, id.
using VehicleRange = thrust::tuple<int, int, int>;

/// Represents search context: data passed through main operators.
struct SearchContext final {
  const Context context;
  int base;
  int customer;
};

/// Represents insertion context: data used to estimate insertion cost.
struct InsertionContext final {
  /// Base index.
  int base;
  /// Task where insertion range starts.
  int from;
  /// Task where insertion range ends.
  int to;
  /// Used vehicle.
  int vehicle;
  /// Customer to be inserted
  int customer;
};

/// Restores vehicle state from insertion context.
EXEC_UNIT inline Transition::State restore_state(const Context& context,
                                                 const InsertionContext& insertion) {
  auto index = insertion.base + thrust::max(insertion.from - 1, 0);
  if (context.tasks.vehicles[index] != insertion.vehicle) index = 0;

  int capacity = index == 0
                   ? static_cast<int>(context.problem.resources.capacities[insertion.vehicle])
                   : static_cast<int>(context.tasks.capacities[index]);

  int time = index == 0 ? 0 : context.tasks.times[index];

  return Transition::State{context.tasks.ids[index], capacity, time};
}

/// Finds next random customer to serve.
struct find_random_customer final {
  __host__ __device__ explicit find_random_customer(const Tasks::Shadow tasks) :
    tasks(tasks), dist(0, tasks.customers), rng() {}

  __host__ __device__ int operator()() {
    auto start = dist(rng);
    auto customer = start;
    bool increment = start % 2 == 0;

    do {
      Plan plan = tasks.plan[customer];

      if (!plan.isAssigned()) return customer;

      // try to find next customer
      if (increment)
        customer = customer == tasks.customers ? 1 : customer + 1;
      else
        customer = customer == 0 ? tasks.customers - 1 : customer - 1;
    } while (customer != start);

    return -1;
  }

private:
  const Tasks::Shadow tasks;
  thrust::uniform_int_distribution<int> dist;
  thrust::minstd_rand rng;
};

/// Estimates insertion to a given arc.
template<typename TransitionOp>
struct estimate_insertion final {
  const SearchContext search;
  const InsertionContext insertion;
  const TransitionOp transitionOp;

  /// @param task Task index from which arc starts.
  EXEC_UNIT InsertionCost operator()(int point) const {
    auto state = restore_state(search.context, insertion);
    float cost = 0;
    for (int i = insertion.from; i != insertion.to; ++i) {
      if (i == point) {
        // TODO add extra
      }

      variant<int, Convolution> customer;
      // TODO: support convolution here?
      customer.set<int>(search.context.tasks.ids[insertion.base + i]);

      auto details = Transition::Details{insertion.base, i, i + 1, customer, insertion.vehicle};
      Transition transition = transitionOp.create(details, state);

      if (!transition.isValid()) return InsertionCost{0, 0};

      cost += transitionOp.estimate(transition);

      state.customer = transition.details.customer.get<int>();
      state.time += transition.delta.duration();
      state.capacity -= transition.delta.demand;
    }

    return InsertionCost{point, cost};
  }
};

///// Compares two arcs using their insertion costs.
struct compare_arcs final {
  EXEC_UNIT InsertionCost operator()(const InsertionCost& left, const InsertionCost& right) const {
    return left.second > right.second ? left : right;
  }
};

/// Finds the "best" arc from single tour where given customer can be inserted.
template<typename TransitionOp>
struct find_best_arc final {
  const SearchContext search;
  const TransitionOp transitionOp;

  EXEC_UNIT InsertionCost operator()(const VehicleRange& range) const {
    if (thrust::get<1>(range) == -1) return InsertionCost{-1, -1};

    int from = thrust::get<0>(range);
    int to = thrust::get<1>(range);
    int vehicle = thrust::get<2>(range);

    return thrust::transform_reduce(
      exec_unit_policy{}, thrust::make_counting_iterator(search.base + 1),
      thrust::make_counting_iterator(to),
      estimate_insertion<TransitionOp>{
        search, {search.base, from, to, vehicle, search.customer}, transitionOp},
      InsertionCost{-1, -1}, compare_arcs{});
  }
};

/// Represents operator which helps to create vehicle ranges without extra memory footprint.
struct create_vehicle_ranges final {
  int last;
  EXEC_UNIT VehicleRange operator()(const VehicleRange& left, const VehicleRange& right) {
    if (thrust::get<2>(left) != thrust::get<2>(right) && thrust::get<1>(left) == -1)
      return {thrust::get<0>(left), thrust::get<0>(right) - 1, thrust::get<2>(left)};

    if (thrust::get<0>(right) == last)
      return {thrust::get<0>(left), thrust::get<0>(right), thrust::get<2>(right)};

    return {thrust::get<1>(left) != -1 ? thrust::get<0>(right) - 1 : thrust::get<0>(left), -1,
            thrust::get<2>(right)};
  }
};

/// Finds the "best" insertion point for given customer inside all tours.
template<typename TransitionOp>
struct find_insertion_point final {
  const TransitionOp transitionOp;

  /// @returns Task index from which to perform transition.
  __host__ __device__ int operator()(const SearchContext& search, int to) {
    auto iterator = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                                 thrust::make_constant_iterator(0),
                                                                 search.context.tasks.vehicles));
    int size = to - 1;

    thrust::inclusive_scan(
      exec_unit_policy{},
      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_counting_iterator(0), thrust::make_constant_iterator(-1),
                           search.context.tasks.vehicles + search.base + 1)),
      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_counting_iterator(size), thrust::make_constant_iterator(1),
                           search.context.tasks.vehicles + search.base + to)),
      vrp::iterators::make_aggregate_output_iterator(
        iterator, find_best_arc<TransitionOp>{search, transitionOp}),
      create_vehicle_ranges{size - 1});

    // TODO get point index as result
    return 0;
  }
};

///// Inserts a new customer in between existing ones.
struct insert_customer final {
  /// @returns Index of last task.
  EXEC_UNIT int operator()(const InsertionContext& context, int point) {
    // TODO
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

template<typename TransitionOp>
void random_insertion<TransitionOp>::operator()(const Context& context, int index, int shift) {
  const auto begin = index * context.problem.size;

  auto transitionOp = TransitionOp(context.problem, context.tasks);
  auto findCustomer = find_random_customer(context.tasks);
  auto findPoint = find_insertion_point<TransitionOp>{transitionOp};
  auto insertCustomer = insert_customer{};

  int to = shift == 0 ? 1 : shift;
  int customer = 0;


  do {
    customer = customer != 0 ? customer : findCustomer();
    // TODO handle initial step where to == 1

    auto point = findPoint(SearchContext{context, begin, customer}, to);

    // TODO
    break;

  } while (to < context.problem.size);

  //  do {
  //    customer = customer != 0 ? customer : findCustomer();
  //    //auto insertion = InsertionContext{begin, to, customer, vehicle};
  //    auto point = findArc(insertion);
  //    if (point > 0) {
  //      from = insertCustomer(insertion, point);
  //      to = from + 1;
  //      customer = 0;
  //    } else {
  //      // NOTE cannot find any further customer to serve within vehicle
  //      if (from == 0 || vehicle == context.problem.resources.vehicles - 1) break;
  //
  //      from = 0;
  //      spawn_vehicle(context.problem, context.tasks, from, ++vehicle);
  //    }
  //  } while (to < context.problem.size);
}

/// NOTE make linker happy.
template class random_insertion<
  TransitionDelegate<vrp::algorithms::transitions::create_transition,
                     vrp::algorithms::costs::calculate_transition_cost,
                     vrp::algorithms::transitions::perform_transition>>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
