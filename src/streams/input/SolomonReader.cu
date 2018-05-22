#include "iterators/CartesianProduct.cu"
#include "models/Resources.hpp"
#include "streams/input/SolomonReader.hpp"

#include <istream>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

using namespace vrp::models;
using namespace vrp::streams;

namespace {
/// Customer defined by: id, x, y, demand, start, end, service
using CustomerData = thrust::tuple<int, int, int, int, int, int, int>;

/// Vehicle type defined by: amount, capacity
using VehicleType = thrust::tuple<int, int>;

using DistanceCalculator = SolomonReader::DistanceCalculator;

/// Skips selected amount of lines from stream.
void skipLines(std::istream& input, int count) {
  for (int i = 0; i < count; ++i)
    input.ignore(std::numeric_limits<std::streamsize>::max(), input.widen('\n'));
}

/// Read resources from stream.
void readResources(std::istream& input, Resources& resources) {
  VehicleType type;

  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  iss >> thrust::get<0>(type) >> thrust::get<1>(type);

  auto count = static_cast<std::size_t>(thrust::get<0>(type));
  auto capacity = thrust::get<1>(type);

  resources.reserve(count);
  resources.capacities.resize(count, capacity);
  resources.distanceCosts.resize(count, 1);
  resources.timeCosts.resize(count, 0);
  resources.waitingCosts.resize(count, 0);
  resources.fixedCosts.resize(count, 0);
  resources.timeLimits.resize(count, std::numeric_limits<int>::max());
}


/// Read customer data from stream.
thrust::host_vector<CustomerData> readCustomerData(std::istream& input) {
  thrust::host_vector<CustomerData> data;
  CustomerData customer;
  while (input) {
    input >> thrust::get<0>(customer) >> thrust::get<1>(customer) >> thrust::get<2>(customer) >>
      thrust::get<3>(customer) >> thrust::get<4>(customer) >> thrust::get<5>(customer) >>
      thrust::get<6>(customer);

    // skip last newlines
    if (!data.empty() && thrust::get<0>(customer) == thrust::get<0>(data[data.size() - 1])) break;

    data.push_back(customer);
  }
  return data;
}

/// Sets customers on GPU.
void setCustomers(const thrust::host_vector<CustomerData>& data, Customers& customers) {
  customers.reserve(data.size());
  thrust::for_each(data.begin(), data.end(), [&](const CustomerData& customer) {
    customers.demands.push_back(thrust::get<3>(customer));
    customers.starts.push_back(thrust::get<4>(customer));
    customers.ends.push_back(thrust::get<5>(customer));
    customers.services.push_back(thrust::get<6>(customer));
  });
}

/// Creates distance matrix.
void setDistances(const thrust::host_vector<CustomerData>& data,
                  thrust::device_vector<float>& distances,
                  const DistanceCalculator& calculator) {
  // TODO move calculations on GPU
  typedef thrust::host_vector<CustomerData>::const_iterator Iterator;
  vrp::iterators::repeated_range<Iterator> repeated(data.begin(), data.end(), data.size());
  vrp::iterators::tiled_range<Iterator> tiled(data.begin(), data.end(), data.size());

  thrust::host_vector<float> hostDist(data.size() * data.size());

  thrust::transform(
    repeated.begin(), repeated.end(), tiled.begin(), hostDist.begin(),
    [&](const CustomerData& left, const CustomerData& right) {
      return thrust::get<0>(left) == thrust::get<0>(right)
               ? 0
               : calculator(thrust::make_tuple(thrust::get<1>(left), thrust::get<2>(left)),
                            thrust::make_tuple(thrust::get<1>(right), thrust::get<2>(right)));
    });

  distances = hostDist;
}

/// Creates durations matrix.
void setDurations(const thrust::host_vector<CustomerData>& data,
                  const thrust::device_vector<float>& distances,
                  thrust::device_vector<int>& durations) {
  durations.resize(data.size() * data.size(), 0);
  thrust::transform(distances.begin(), distances.end(), durations.begin(),
                    thrust::placeholders::_1);
}

void readProblem(std::istream& input, Problem& problem, const DistanceCalculator& calculator) {
  auto data = readCustomerData(input);

  setCustomers(data, problem.customers);
  setDistances(data, problem.routing.distances, calculator);
  setDurations(data, problem.routing.distances, problem.routing.durations);
}

}  // namespace

namespace vrp {
namespace streams {

Problem SolomonReader::read(std::istream& input, const DistanceCalculator& calculator) {
  Problem problem;

  skipLines(input, 4);

  readResources(input, problem.resources);

  skipLines(input, 4);

  readProblem(input, problem, calculator);

  return std::move(problem);
}

}  // namespace streams
}  // namespace vrp
