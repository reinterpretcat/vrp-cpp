#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <algorithm>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::utils;

namespace {

template<typename T>
struct is final {
  const T value;
  bool operator()(const T& other) const { return other == value; }
};

inline void addError(SolutionChecker::Result& result, int individuum, const char* message) {
  result.errors.push_back(std::string("Individuum ") + std::to_string(individuum) + ": " + message);
}

inline void addError(SolutionChecker::Result& result, const char* message) {
  result.errors.push_back(std::string("Solution: ") + message);
}

/// Checks that list of customer ids is correct.
struct check_customers final {
  SolutionChecker::Result& result;
  int index;

  template<typename Iterator>
  void operator()(Iterator begin, Iterator end) {
    typedef typename Iterator::value_type ValueType;

    thrust::device_vector<ValueType> sorted(begin, end);
    thrust::sort(thrust::device, sorted.begin(), sorted.end());

    // ensure bounds
    if (sorted[0] != 0) {
      addError(result, index, "has non zero at depot position.");
      return;
    }

    thrust::device_vector<ValueType> differences(static_cast<size_t>(thrust::distance(begin, end)));
    thrust::adjacent_difference(thrust::device, sorted.begin(), sorted.end(), differences.begin());
    if (!thrust::all_of(thrust::device, differences.begin() + 1, differences.end(),
                        is<ValueType>{1}))
      addError(result, index, "has non unique customer ids.");
  }
};

/// Checks vehicles allocation.
struct check_vehicles final {
  const Solution& solution;
  SolutionChecker::Result& result;
  int index;

  template<typename Iterator>
  void operator()(Iterator begin, Iterator end) {
    typedef typename Iterator::value_type ValueType;

    // check that vehicles are allocated sequentially
    thrust::device_vector<ValueType> sorted(begin, end);
    thrust::sort(thrust::device, sorted.begin(), sorted.end());

    if (sorted[0] != 0) addError(result, index, "unexpected first vehicle.");

    if (sorted[sorted.size() - 1] + 1 > solution.problem.resources.capacities.size())
      addError(result, index, "has extra vehicles.");

    if (!thrust::equal(thrust::device, begin, end, sorted.begin()))
      addError(result, index, "has wrong vehicle order.");
  }
};

struct verify_tasks final {
  SolutionChecker::Result& result;

  void operator()(const Tasks& left, const Tasks& right, int begin, int end) const {
    if (!thrust::equal(thrust::device, left.ids.begin() + begin, left.ids.begin() + end,
                       right.ids.begin() + begin))
      addError(result, "unexpected ids!");

    if (!thrust::equal(thrust::device, left.vehicles.begin() + begin, left.vehicles.begin() + end,
                       right.vehicles.begin() + begin))
      addError(result, "unexpected vehicles!");

    if (!thrust::equal(thrust::device, left.costs.begin() + begin, left.costs.begin() + end,
                       right.costs.begin() + begin))
      addError(result, "unexpected costs!");

    if (!thrust::equal(thrust::device, left.capacities.begin() + begin,
                       left.capacities.begin() + end, right.capacities.begin() + begin))
      addError(result, "unexpected capacities!");

    if (!thrust::equal(thrust::device, left.times.begin() + begin, left.times.begin() + end,
                       right.times.begin() + begin))
      addError(result, "unexpected times!");

    if (!thrust::equal(thrust::device, left.plan.begin() + begin, left.plan.begin() + end,
                       right.plan.begin() + begin))
      addError(result, "unexpected plan!");
  }
};

/// Check all tours.
struct check_tours final {
  const Solution& solution;
  SolutionChecker::Result& result;
  int index;

  void operator()(int begin, int end) {
    auto errors = result.errors.size();
    auto tasks =
      Tasks(solution.problem.size(), solution.tasks.population() * solution.problem.size());

    auto factory = create_transition(solution.problem.getShadow(), tasks.getShadow());
    auto executor = perform_transition(solution.problem.getShadow(), tasks.getShadow());
    auto costs = calculate_transition_cost(solution.problem.resources.getShadow());

    int from = 0;
    int to = from + 1;
    int vehicle = 0;
    int last = end - begin;

    createDepotTask(tasks, begin + from);

    do {
      auto wrapped = device_variant<int, Convolution>();
      wrapped.set<int>(solution.tasks.ids[begin + to]);
      auto transition = factory({begin, from, to, wrapped, vehicle});

      if (!transition.isValid()) {
        // do we have to use next vehicle?
        if (solution.tasks.vehicles[begin + to] == vehicle + 1) {
          spawnNewVehicle(tasks, begin + to, ++vehicle);
          from = 0;
          continue;
        }

        addError(result, index,
                 (std::string("Cannot serve customer at ") + std::to_string(to) + " task").c_str());
        return;
      }

      executor(transition, costs(transition));

      ensureConstraints(tasks, begin + to, vehicle);

      from = to++;

    } while (to < last);

    verify_tasks{result}(solution.tasks, tasks, begin, end);

    dumpIfErrors(solution.tasks, tasks, index, errors);
  }

private:
  /// Creates depot task.
  void createDepotTask(Tasks& tasks, int task) {
    const int depot = 0;
    const int vehicle = 0;

    tasks.ids[task] = depot;
    tasks.times[task] = solution.problem.customers.starts[0];
    tasks.capacities[task] = solution.problem.resources.capacities[vehicle];
    tasks.vehicles[task] = vehicle;
    tasks.costs[task] = solution.problem.resources.fixedCosts[vehicle];
    tasks.plan[task] = Plan::assign();
  }

  /// Spawns new vehicle.
  void spawnNewVehicle(Tasks& tasks, int task, int vehicle) {
    tasks.times[task] = solution.problem.customers.starts[0];
    tasks.capacities[task] = solution.problem.resources.capacities[vehicle];
    tasks.costs[task] = solution.problem.resources.fixedCosts[vehicle];
  }

  /// Ensures that vehicle constraints are not violated.
  void ensureConstraints(Tasks& tasks, int task, int vehicle) {
    int customer = tasks.ids[task];
    int arrivalTime = tasks.times[task] - solution.problem.customers.services[customer];
    int returnTime =
      tasks.times[task] + solution.problem.routing.durations[index * solution.problem.size()];

    if (tasks.capacities[task] < 0)
      addError(result, (std::to_string(tasks.capacities[task]) +
                        std::string(" requested capacity exceeds capacity of vehicle ") +
                        std::to_string(vehicle))
                         .c_str());

    if (arrivalTime < solution.problem.customers.starts[customer] ||
        arrivalTime > solution.problem.customers.ends[customer])
      addError(result,
               (std::to_string(arrivalTime) +
                std::string(" as arrival time violates time window constraint of customer ") +
                std::to_string(customer))
                 .c_str());

    if (tasks.times[task] + returnTime > solution.problem.resources.timeLimits[vehicle])
      addError(result, (std::to_string(tasks.times[task]) +
                        std::string(" time violates return constraint of vehicle ") +
                        std::to_string(vehicle))
                         .c_str());

    if (returnTime > solution.problem.customers.ends[0])
      addError(result, (std::to_string(tasks.times[task]) +
                        std::string(" time violates end time window constraint of depot "))
                         .c_str());
  }
  /// Dumps tasks into std out if new errors occurred.
  void dumpIfErrors(const Tasks& left, const Tasks& right, int index, int errors) {
    if (errors < result.errors.size()) {
      std::cout << "Errors in " << index << "\n";
      MatrixTextWriter::write(std::cout, left);
      MatrixTextWriter::write(std::cout, right);
    }
  }
};

}  // namespace

namespace vrp {
namespace utils {

SolutionChecker::Result SolutionChecker::check(const Solution& solution) {
  auto result = SolutionChecker::Result();

  std::for_each(thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(solution.tasks.population()), [&](int index) {
                  auto begin = index * solution.problem.size();
                  auto end = begin + solution.problem.size();

                  check_customers{result, index}(solution.tasks.ids.begin() + begin,
                                                 solution.tasks.ids.begin() + end);

                  check_vehicles{solution, result, index}(solution.tasks.vehicles.begin() + begin,
                                                          solution.tasks.vehicles.begin() + end);

                  check_tours{solution, result, index}(begin, end);
                });

  return result;
}


}  // namespace utils
}  // namespace vrp