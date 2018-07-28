#include "algorithms/heuristics/Operators.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "runtime/UniquePointer.hpp"
#include "iterators/Aggregates.hpp"

#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/pair.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Specifies vehicle range: start, end, id.
using VehicleRange = thrust::tuple<int, int, int>;

/// Represents search context: data passed through main operators.
struct SearchContext final {
  const Context context;
  int base;
  int last;
  int customer;
};

/// Represents insertion context: data used to estimate insertion cost.
struct InsertionContext final {
  /// Base index.
  int base;
  /// Last index.
  int last;
  /// Task where insertion range starts.
  int from;
  /// Task where insertion range ends.
  int to;
  /// Used vehicle.
  int vehicle;
  /// Customer to be inserted
  int customer;
};

struct Result final {
  /// Task where vehicle range starts.
  int from;
  /// Task where vehicle range ends.
  int to;
  /// Insertion Point
  int point;
  /// Estimated insertion cost
  float cost;
};

inline Result create_invalid_data() {
  return {-1, -1, -1, __FLT_MAX__};
}

/// Finds next random customer to serve.
struct find_random_customer final {
  ANY_EXEC_UNIT explicit find_random_customer(const Tasks::Shadow tasks) :
    tasks(tasks), dist(0, tasks.customers), rng() {}

  ANY_EXEC_UNIT int operator()() {
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
  EXEC_UNIT Result operator()(int point) const {
    float cost = 0;
    Transition::State state;

    restore(point, state, cost);

    if (!update(insertion.customer, point, state, cost))
      return create_invalid_data();

    for (int i = point + 1; i <= insertion.to; ++i) {
      auto customer = search.context.tasks.ids[insertion.base + i];
      if (!update(customer, i, state, cost))
        return create_invalid_data();
    }

    return Result{insertion.from, insertion.to, point, cost};
  }
 private:

  /// Restores state before insertion point.
  EXEC_UNIT void restore(int point, Transition::State& state, float& cost) const {
    const auto& context = search.context;

    auto index = insertion.base + point;//thrust::max(point - 1, 0);
    //if (context.tasks.vehicles[index] != insertion.vehicle) index = 0;

    int capacity = index == 0
                   ? context.problem.resources.capacities[insertion.vehicle]
                   : context.tasks.capacities[index];

    int time = index == 0 ? 0 : context.tasks.times[index];

    cost = index == 0 ? 0 : context.tasks.costs[index];

    state.customer =context.tasks.ids[index];
    state.capacity = capacity;
    state.time = time;
  }

  /// Updates state within new customer.
  EXEC_UNIT bool update(int id, int task, Transition::State& state, float& cost) const {
    variant<int, Convolution> customer;
    // TODO: support convolution here?
    customer.set<int>(id);

    auto details = Transition::Details{insertion.base, task, task + 1, customer, insertion.vehicle};
    Transition transition = transitionOp.create(details, state);

    if (!transition.isValid()) return false;

    cost += transitionOp.estimate(transition);

    state.customer = transition.details.customer.get<int>();
    state.time += transition.delta.duration();
    state.capacity -= transition.delta.demand;

    return true;
  }
};

///// Compares two arcs using their insertion costs.
struct compare_arcs final {
  EXEC_UNIT Result operator()(const Result& left, const Result& right) const {
    return left.cost > right.cost ? left : right;
  }

  EXEC_UNIT bool operator()(const Result* left, const Result* right) const {
    return left->cost > right->cost;
  }
};

/// Finds the "best" arc from single tour where given customer can be inserted.
template<typename TransitionOp>
struct find_best_arc final {
  const SearchContext search;
  const TransitionOp transitionOp;
  vector_ptr<Result> results;

  EXEC_UNIT Result operator()(const VehicleRange& range) const {
    if (thrust::get<1>(range) == -1) return create_invalid_data();

    int from = thrust::get<0>(range);
    int to = thrust::get<1>(range);
    int vehicle = thrust::get<2>(range);

    results[vehicle] = thrust::transform_reduce(
      exec_unit_policy{},
      thrust::make_counting_iterator(search.base + from),
      thrust::make_counting_iterator(search.base + to + 1),
      estimate_insertion<TransitionOp>{
        search, {search.base, /* TODO */ 0, from, to, vehicle, search.customer}, transitionOp},
      Result{from, to, -1, -1}, compare_arcs{});

    return {};
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
  unique_ptr<vector_ptr<Result>> results;

  /// @returns Task index from which to perform transition.
  ANY_EXEC_UNIT Result operator()(const SearchContext& search, int vehicle) {

  auto iterator = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                                 thrust::make_constant_iterator(0),
                                                                 search.context.tasks.vehicles));
    // first iteration
    if (search.last == 1) return Result {0, search.last, 1, 0};

    thrust::inclusive_scan(
      exec_unit_policy{},
      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_counting_iterator(0),
                           thrust::make_constant_iterator(-1),
                           search.context.tasks.vehicles + search.base)),
      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_counting_iterator(search.last),
                            thrust::make_constant_iterator(1),
                           search.context.tasks.vehicles + search.base + search.last)),
      vrp::iterators::make_aggregate_output_iterator(
        iterator, find_best_arc<TransitionOp>{search, transitionOp, *results}),
      create_vehicle_ranges{search.last - 1});

    auto ggg = thrust::max_element(
        exec_unit_policy{},
        results.get(),
        results.get() + search.context.tasks.vehicles[search.last] + 1,
        compare_arcs{}
    );

    // TODO
    return **ggg;
  }
};

///// Inserts a new customer in between existing ones.
template<typename TransitionOp>
struct insert_customer final {
  const TransitionOp transitionOp;

  /// @returns Index of last task.
  EXEC_UNIT int operator()(const SearchContext& search, const Result& data) {
    // insert as last
    if (data.to == search.last) {
      variant<int, Convolution> customer;
      customer.set<int>(search.customer);

      int vehicle = search.context.tasks.vehicles[data.from];

      auto details = Transition::Details{search.base, data.from, data.to,
                                         customer, vehicle};
      auto transition = transitionOp.create(details);
      auto cost = transitionOp.estimate(transition);
      return transitionOp.perform(transition, cost);
    }
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
  auto findPoint = find_insertion_point<TransitionOp>{transitionOp,
                                                      make_unique_ptr_data<Result>(context.problem.size)};
  auto insertCustomer = insert_customer<TransitionOp>{transitionOp};

  int to = shift == 0 ? 1 : shift;
  int customer = 0;
  int vehicle = context.tasks.vehicles[to - 1];

  bool stop = false;

  do {
    customer = customer != 0 ? customer : findCustomer();

    auto search = SearchContext{context, begin, to, customer};
    auto insertion = findPoint(search, vehicle);

    // allocate new  vehicle if estimation fails to insert customer
    if (insertion.point == -1) {
      ++vehicle;
      continue;
    }
    else customer = 0;

    to = insertCustomer(search, insertion) + 1;


    // TODO remove
    if (stop) break;
    stop = true;

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
