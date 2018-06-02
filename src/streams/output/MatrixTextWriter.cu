#include "../../models/Tasks.hpp"
#include "algorithms/costs/SolutionCosts.hpp"
#include "models/Problem.hpp"
#include "streams/output/MatrixTextWriter.hpp"

#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

using namespace vrp::algorithms::costs;
using namespace vrp::models;
using namespace vrp::streams;

namespace {
const int ItemSize = 4;

/// Prints item with additional formatting.
struct print_one final {
  std::ostream& stream;
  int customers;

  template<typename T>
  __host__ void operator()(const thrust::tuple<int, T> item) {
    stream << (thrust::get<0>(item) % customers == 0 ? '\n' : ',') << std::setfill(' ')
           << std::setw(ItemSize) << std::fixed << std::setprecision(0) << thrust::get<1>(item);
  }
};

/// Writes vectorized data into stream.
template<typename T>
void writeVector(std::ostream& stream, const thrust::device_vector<T>& data, int populationSize) {
  thrust::host_vector<T> hData(data.begin(), data.end());
  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), hData.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_counting_iterator(static_cast<int>(data.size())), hData.end())),
    print_one{stream, static_cast<int>(data.size() / populationSize)});
}

/// Writes total costs.
void writeCosts(std::ostream& stream, Solution& solution) {
  stream << thrust::transform_reduce(
    thrust::host, thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(solution.tasks.population()),
    [&](const int i) {
      return std::to_string(static_cast<int>(calculate_total_cost()(solution, i)));
    },
    std::string(""),
    [](const std::string& result, const std::string& item) {
      return result + (result.empty() ? "\n  " : ", ") + item;
    });
}

}  // namespace

void MatrixTextWriter::write(std::ostream& out, const vrp::models::Solution& solution) {
  writeCosts(out << "\ntotal costs: ", const_cast<Solution&>(solution));
  writeVector(out << "\ncustomers:   ", solution.tasks.ids, solution.tasks.population());
  writeVector(out << "\nvehicles:    ", solution.tasks.vehicles, solution.tasks.population());
  writeVector(out << "\ncosts:       ", solution.tasks.costs, solution.tasks.population());
  writeVector(out << "\ncapacities:  ", solution.tasks.capacities, solution.tasks.population());
  writeVector(out << "\ntimes:       ", solution.tasks.times, solution.tasks.population());
}
