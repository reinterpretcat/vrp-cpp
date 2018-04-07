#ifndef VRP_STREAMS_SOLOMONREADER_HPP
#define VRP_STREAMS_SOLOMONREADER_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "iterators/CartesianProduct.cu"

#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <istream>
#include <sstream>

namespace vrp {
namespace streams {

/// Reads classical VRP instances in classical format defined by Solomon.
template <typename T>
class SolomonReader final {

  /// Customer defined by: id, x, y, demand, start, end, service
  using CustomerData = thrust::tuple<int, int, int, int, int, int, int>;

  /// Vehicle type defined by: amount, capacity
  using VehicleType = thrust::tuple<int, int>;

  /// Calculates distance between two customers using T.
  struct DistanceCalculator {
    __host__ __device__
    float operator()(const CustomerData &left, const CustomerData &right) {
      return thrust::get<0>(left) == thrust::get<0>(right)
             ? 0
             : T()(thrust::make_tuple(thrust::get<1>(left), thrust::get<2>(left)),
                   thrust::make_tuple(thrust::get<1>(right), thrust::get<2>(right)));
    }
  };

 public:
  /// Creates VRP problem and resources from input stream.
  static vrp::models::Problem read(std::istream &input) {
    vrp::models::Problem problem;

    skipLines(input, 4);

    readResources(input, problem.resources);

    skipLines(input, 4);

    readProblem(input, problem);

    return std::move(problem);
  }

 private:

  /// Skips selected amount of lines from stream.
  static void skipLines(std::istream &input, int count) {
    for (int i = 0; i < count; ++i)
      input.ignore(std::numeric_limits<std::streamsize>::max(), input.widen('\n'));
  }

  /// Read resources from stream.
  static void readResources(std::istream &input, vrp::models::Resources &resources) {
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

  static void readProblem(std::istream &input, vrp::models::Problem &problem) {
    auto data = readCustomerData(input);

    setCustomers(data, problem.customers);
    setDistances(data, problem.routing.distances);
    setDurations(data, problem.routing.distances, problem.routing.durations);
  }

  /// Read customer data from stream.
  static thrust::host_vector<CustomerData> readCustomerData(std::istream &input) {
    thrust::host_vector<CustomerData> data;
    CustomerData customer;
    while (input) {
      input >> thrust::get<0>(customer) >> thrust::get<1>(customer)
            >> thrust::get<2>(customer) >> thrust::get<3>(customer)
            >> thrust::get<4>(customer) >> thrust::get<5>(customer)
            >> thrust::get<6>(customer);

      // skip last newlines
      if (!data.empty() && thrust::get<0>(customer) == thrust::get<0>(data[data.size() - 1]))
        break;

      data.push_back(customer);
    }
    return data;
  }

  /// Sets customers on GPU.
  static void setCustomers(const thrust::host_vector<CustomerData> &data,
                           vrp::models::Customers &customers) {
    customers.reserve(data.size());
    thrust::for_each(data.begin(), data.end(),
                     [&](const CustomerData &customer) {
                       customers.demands.push_back(thrust::get<3>(customer));
                       customers.starts.push_back(thrust::get<4>(customer));
                       customers.ends.push_back(thrust::get<5>(customer));
                       customers.services.push_back(thrust::get<6>(customer));
                     });
  }

  /// Creates distance matrix.
  static void setDistances(const thrust::host_vector<CustomerData> &data,
                           thrust::device_vector<float> &distances) {
    // TODO move calculations on GPU
    typedef thrust::host_vector<CustomerData>::const_iterator Iterator;
    vrp::iterators::repeated_range<Iterator> repeated(data.begin(), data.end(), data.size());
    vrp::iterators::tiled_range<Iterator> tiled(data.begin(), data.end(), data.size());

    thrust::host_vector<float> hostDist(data.size() * data.size());

    thrust::transform(repeated.begin(), repeated.end(),
                      tiled.begin(),
                      hostDist.begin(),
                      DistanceCalculator());

    distances = hostDist;
  }

  /// Creates durations matrix.
  static void setDurations(const thrust::host_vector<CustomerData> &data,
                           const thrust::device_vector<float> &distances,
                           thrust::device_vector<int> &durations) {
    durations.resize(data.size() * data.size(), 0);
    thrust::transform(distances.begin(), distances.end(),
                      durations.begin(),
                      thrust::placeholders::_1);
  }
};

}
}

#endif //VRP_STREAMS_SOLOMONREADER_HPP
