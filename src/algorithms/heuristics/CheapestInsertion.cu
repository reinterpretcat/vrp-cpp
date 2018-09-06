#include "algorithms/common/Tours.hpp"
#include "algorithms/heuristics/CheapestInsertion.hpp"
#include "algorithms/heuristics/ConvolutionInsertion.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "iterators/Aggregates.hpp"
#include "runtime/UniquePointer.hpp"

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>

using namespace vrp::algorithms::common;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Alias for customer variant.
using Customer = variant<int, Convolution>;

/// Represents search context: data passed through main operators.
struct SearchContext final {
  Context context;
  Customer customer;
  int base;
  int last;
};

/// Stores data used to estimate insertion.
struct InsertionData final {
  /// Task where vehicle range starts.
  int from;
  /// Task where vehicle range ends.
  int to;
  /// Used vehicle.
  int vehicle;
  /// Customer to be inserted.
  Customer customer;
};

/// Stores insertion result.
struct InsertionResult final {
  /// Contains insertion data.
  InsertionData data;
  /// Insertion Point
  int point;
  /// Estimated insertion cost
  float cost;
};

/// Creates invalid insertion result.
EXEC_UNIT
inline InsertionResult create_invalid_data() { return {{}, -1, __FLT_MAX__}; }

/// Checks whether customer is depot (id=0).
EXEC_UNIT
inline bool isDepot(const Customer& customer) {
  return customer.is<int>() && customer.get<int>() == 0;
}

/// Creates state for current task
EXEC_UNIT Transition::State createState(const SearchContext& search, int task) {
  auto index = search.base + task;
  return {search.context.tasks.ids[index], search.context.tasks.capacities[index],
          search.context.tasks.times[index]};
}

template<typename TransitionOp>
struct state_processor final {
  const SearchContext search;
  const TransitionOp transitionOp;

  Transition::State state;
  float cost;

  /// Restores state before insertion point.
  EXEC_UNIT void restore(int point, int base, int vehicle) {
    const auto& context = search.context;

    auto index = base + point;

    int capacity = point == 0 ? static_cast<int>(context.problem.resources.capacities[vehicle])
                              : context.tasks.capacities[index];

    int time = point == 0 ? 0 : context.tasks.times[index];

    cost = point == 0 ? 0 : context.tasks.costs[index];

    state.customer = context.tasks.ids[index];
    state.capacity = capacity;
    state.time = time;
  }

  /// Analyzes transition and updates state.
  EXEC_UNIT int analyze(Customer customer, int task, int base, int vehicle) {
    auto details = Transition::Details{base, task, task + 1, customer, vehicle};
    Transition transition = transitionOp.create(details, state);

    if (!transition.isValid()) return -1;
    // TODO estimate does not consider convolution
    cost += transitionOp.estimate(transition);
    return transitionOp.analyze(transition, state);
  }

  EXEC_UNIT int perform(Customer customer, int task, int base, int vehicle) {
    auto details = Transition::Details{search.base, task, task + 1, customer, vehicle};
    Transition transition = transitionOp.create(details, state);
    auto cost = transitionOp.estimate(transition);
    auto current = transitionOp.perform(transition, cost);

    // NOTE analyze cannot be used here due to TW
    state = createState(search, current);

    return current;
  }

  EXEC_UNIT Customer customer(int task) {
    return Customer::create<int>(search.context.tasks.ids[task]);
  }

  EXEC_UNIT float costs(int task) { return search.context.tasks.costs[task]; }
};

/// Estimates insertion to a given arc.
template<typename TransitionOp>
struct estimate_insertion final {
  const InsertionData data;
  state_processor<TransitionOp> stateOp;
  int base;

  /// @param task Task index from which arc starts.
  EXEC_UNIT InsertionResult operator()(int point) {
    int vehicle = data.vehicle;
    float cost = stateOp.costs(base + data.to);

    stateOp.restore(point, base, vehicle);

    if (stateOp.analyze(data.customer, point, base, vehicle) < 0) return create_invalid_data();

    for (int i = point + 1; i <= data.to; ++i) {
      auto customer = stateOp.customer(base + i);
      if (stateOp.analyze(customer, i, base, vehicle) < 0) return create_invalid_data();
    }

    return InsertionResult{
      {data.from, data.to, vehicle, data.customer}, point, stateOp.cost - cost};
  }
};

///// Compares two arcs using their insertion costs.
struct compare_arcs_value final {
  EXEC_UNIT InsertionResult operator()(const InsertionResult& left,
                                       const InsertionResult& right) const {
    return left.cost < right.cost ? left : right;
  }
};

///// Compares two arcs using their insertion costs.
struct compare_arcs_logical final {
  EXEC_UNIT bool operator()(const InsertionResult& left, const InsertionResult& right) const {
    return left.cost < right.cost;
  }
};

/// Finds the "best" arc from single tour where given customer can be inserted.
template<typename TransitionOp>
struct find_best_arc final {
  const SearchContext search;
  const TransitionOp transitionOp;
  vector_ptr<InsertionResult> results;

  EXEC_UNIT InsertionResult operator()(const thrust::tuple<int, int, int, int>& range) const {
    if (thrust::get<0>(range) == -1) return create_invalid_data();

    int from = thrust::get<0>(range);
    int to = thrust::get<1>(range);
    int vehicle = thrust::get<2>(range);

    auto data = InsertionData{from, to, vehicle, search.customer};

    results[vehicle] = thrust::transform_reduce(
      exec_unit_policy{}, thrust::make_counting_iterator(from),
      thrust::make_counting_iterator(to + 1),

      estimate_insertion<TransitionOp>{
        data, state_processor<TransitionOp>{search, transitionOp, {}, 0}, search.base},

      InsertionResult{data, -1, __FLT_MAX__}, compare_arcs_value{});

    return {};
  }
};

/// Finds the "best" insertion point for given customer inside all tours.
template<typename TransitionOp>
struct find_insertion_point final {
  const TransitionOp transitionOp;
  unique_ptr<vector_ptr<InsertionResult>> results;

  /// @returns Task index from which to perform transition.
  EXEC_UNIT InsertionResult operator()(const SearchContext& search, int vehicle) {
    auto lastVehicle = search.context.tasks.vehicles[search.base + search.last - 1];

    // first customer in tour
    if (search.last == 1 || lastVehicle != vehicle)
      return InsertionResult{{0, search.last, vehicle, search.customer}, search.last, 0};

    auto iterator = vrp::iterators::make_aggregate_output_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_counting_iterator(0), thrust::make_constant_iterator(0),
                           search.context.tasks.vehicles, thrust::make_constant_iterator(0))),
      find_best_arc<TransitionOp>{search, transitionOp, *results.get()});

    find_tours<decltype(search.context.tasks.vehicles), decltype(iterator)>{lastVehicle}(
      search.context.tasks.vehicles + search.base,
      search.context.tasks.vehicles + search.base + search.last, iterator);

    return *thrust::min_element(exec_unit_policy{}, *results.get(),
                                *results.get() + lastVehicle + 1, compare_arcs_logical{});
  }
};

///// Inserts a new customer in between existing ones.
template<typename TransitionOp>
struct insert_customer final {
  const TransitionOp transitionOp;

  /// @returns Index of last task.
  EXEC_UNIT int operator()(const SearchContext& search, const InsertionResult& data) {
    return data.point == search.last ? insertLast(search, data) : insertInBetween(search, data);
  }

private:
  /// Inserts new customer as last.
  EXEC_UNIT int insertLast(const SearchContext& search, const InsertionResult& result) {
    auto details = Transition::Details{search.base, result.data.from, result.data.to,
                                       result.data.customer, result.data.vehicle};
    auto transition = transitionOp.create(details);
    auto cost = transitionOp.estimate(transition);
    return transitionOp.perform(transition, cost);
  }

  /// Inserts new customer in single tour.
  EXEC_UNIT int insertInBetween(const SearchContext& search, const InsertionResult& result) {
    // prepare existing data for insertion
    auto shift = shiftData(search, result);

    // prepare new customer
    auto stateOp = state_processor<TransitionOp>{search, transitionOp};
    stateOp.restore(result.point, search.base, result.data.vehicle);

    // insert and recalculate rest of affected tour
    int last = result.point;
    for (int i = result.point; i <= result.data.to; ++i) {
      auto customer =
        i == result.point ? result.data.customer : stateOp.customer(search.base + i + shift);
      last = stateOp.perform(customer, last, search.base, result.data.vehicle);
    }

    return search.last + shift - 1;
  }

  /// Shifts data to right in order to allow insert new customer(-s).
  EXEC_UNIT int shiftData(const SearchContext& search, const InsertionResult& result) {
    int shift = getShiftCount(search, result);

    int begin = search.base + result.point;
    int end = search.base + search.last + shift - 1;
    auto tasks = search.context.tasks;

    // shift everything to the right
    shiftRight(tasks.ids + begin, tasks.ids + end, shift);
    shiftRight(tasks.costs + begin, tasks.costs + end, shift);
    shiftRight(tasks.vehicles + begin, tasks.vehicles + end, shift);
    shiftRight(tasks.capacities + begin, tasks.capacities + end, shift);
    shiftRight(tasks.times + begin, tasks.times + end, shift);

    return shift;
  }

  /// Shifts to the right all data.
  template<typename T>
  EXEC_UNIT void shiftRight(T begin, T end, int shift) {
    for (int i = 0; i < shift; ++i) {
      for (auto iter = end - 1; iter >= begin; --iter) {
        *(iter + 1) = *iter;
      }
    }
  }

  /// Calculates how much data should be shifted to the right.
  EXEC_UNIT int getShiftCount(const SearchContext& search, const InsertionResult& result) const {
    if (search.customer.is<int>()) return 1;

    auto details = Transition::Details{search.base, result.point, result.point + 1,
                                       result.data.customer, result.data.vehicle};
    auto transition = transitionOp.create(details);
    auto state = createState(search, details.from);
    auto next = transitionOp.analyze(transition, state);

    return next - details.from;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

template<typename TransitionOp, typename CustomerSelector>
EXEC_UNIT void cheapest_insertion<TransitionOp, CustomerSelector>::
operator()(const Context& context, int index, int shift) {
  const auto begin = index * context.problem.size;

  auto transitionOp = TransitionOp(context.problem, context.tasks);
  auto findCustomer = CustomerSelector(context, begin);
  auto findPoint = find_insertion_point<TransitionOp>{
    transitionOp, make_unique_ptr_data<InsertionResult>(context.problem.size)};
  auto insertCustomer = insert_customer<TransitionOp>{transitionOp};

  auto customer = Customer::create<int>(0);
  int to = shift == 0 ? 1 : shift + 1;
  int vehicle = context.tasks.vehicles[to - 1];

  while (to < context.problem.size) {
    customer = isDepot(customer) ? findCustomer() : customer;

    auto search = SearchContext{context, customer, begin, to};
    auto insertion = findPoint(search, vehicle);

    // allocate new vehicle if estimation fails to insert customer
    if (insertion.point == -1) {
      ++vehicle;
      continue;
    }

    to = insertCustomer(search, insertion) + 1;

    customer = Customer::create<int>(0);
  }
}

/// NOTE make linker happy.
template class cheapest_insertion<TransitionOperator, find_random_customer>;
template class cheapest_insertion<TransitionOperator, find_convolution_customer>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
