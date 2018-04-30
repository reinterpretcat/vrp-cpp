#include "algorithms/Costs.cu"
#include "models/Problem.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "../../models/Tasks.hpp"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>

#include <iomanip>

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::streams;

namespace {
const int ItemSize = 4;

/// Prints item with additional formatting.
struct PrintOne final {
  std::ostream& stream;
  int customers;

  template <typename T>
  __host__
  void operator()(const thrust::tuple<int,T> item) {
    stream
        << (thrust::get<0>(item) % customers == 0 ? '\n' : ',')
        << std::setfill(' ') << std::setw(ItemSize)
        << std::fixed << std::setprecision(0)
        << thrust::get<1>(item);
  }
};

/// Writes vectorized data into stream.
template <typename T>
void writeVector(std::ostream& stream, const thrust::device_vector<T>& data, int populationSize) {
  thrust::host_vector<T> hData (data.begin(), data.end());
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(0),
          hData.begin())
      ),
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(static_cast<int>(data.size())),
          hData.end())
      ),
      PrintOne { stream, static_cast<int>(data.size() / populationSize) }
  );
  //thrust::copy(data.begin(), data.end(), std::ostream_iterator<T>(stream, ","));
}

/// Writes total costs.
void writeCosts(std::ostream& stream, const Problem &problem, Tasks &tasks) {
  stream << thrust::transform_reduce(
      thrust::host,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(tasks.population()),
      [&](const int i) {
        return std::to_string(static_cast<int>(calculate_total_cost()(problem, tasks, i)));
      },
      std::string(""),
      [](const std::string &result, const std::string &item) {
        return result + (result.empty() ? "\n  " : ", ") + item;
      }
  );
}

}

void TextWriter::write(std::ostream &out, const Problem &problem, const Tasks &tasks) {
  writeCosts( out << "\ntotal costs: ", problem, const_cast<Tasks&>(tasks));
  writeVector(out << "\ncustomers:   ", tasks.ids, tasks.population());
  writeVector(out << "\nvehicles:    ", tasks.vehicles, tasks.population());
  writeVector(out << "\ncosts:       ", tasks.costs, tasks.population());
  writeVector(out << "\ncapacities:  ", tasks.capacities, tasks.population());
  writeVector(out << "\ntimes:       ", tasks.times, tasks.population());
}
