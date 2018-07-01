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

template<typename T>
void write(std::ostream& stream, const T& value) {
  stream << value;
}

template<>
void write(std::ostream& stream, const Plan& plan) {
  if (plan.hasConvolution())
    stream << std::setw(ItemSize - 1) << 'c' << plan.convolution();
  else
    stream << plan.isAssigned();
}

/// Prints item with additional formatting.
struct print_one final {
  std::ostream& stream;
  int customers;


  template<typename T>
  __host__ void operator()(const thrust::tuple<int, T> item) {
    stream << (thrust::get<0>(item) % customers == 0 ? '\n' : ',') << std::setfill(' ')
           << std::setw(ItemSize) << std::fixed << std::setprecision(0);
    write(stream, thrust::get<1>(item));
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
  write(out, solution.tasks);
}

void MatrixTextWriter::write(std::ostream& out, const vrp::models::Tasks& tasks) {
  writeVector(out << "\ncustomers:   ", tasks.ids, tasks.population());
  writeVector(out << "\nvehicles:    ", tasks.vehicles, tasks.population());
  writeVector(out << "\ncosts:       ", tasks.costs, tasks.population());
  writeVector(out << "\ncapacities:  ", tasks.capacities, tasks.population());
  writeVector(out << "\ntimes:       ", tasks.times, tasks.population());
  writeVector(out << "\nplan:        ", tasks.plan, tasks.population());
  out << "\n";
}
