#ifndef VRP_INPUT_SOLOMONREADER_HPP
#define VRP_INPUT_SOLOMONREADER_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "utils/CartesianProduct.cu"
#include "utils/StreamUtils.hpp"

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <istream>
#include <sstream>

namespace vrp {
namespace input {

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
  static void read(std::istream &input,
                   vrp::models::Problem &problem) {

    skipLines(input, 4);

    readResources(input, problem.resources);

    skipLines(input, 4);

    readProblem(input, problem);
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

    resources.vehicleAvalabilities.push_back(thrust::get<0>(type));
    resources.vehicleCapacities.push_back(thrust::get<1>(type));
    resources.vehicleDistanceCosts.push_back(1);
    // NOTE not specified
    resources.vehicleTimeCosts.push_back(0);
    resources.vehicleWaitingCosts.push_back(0);
    resources.vehicleTimeLimits.push_back(0);
  }

  static void readProblem(std::istream &input, vrp::models::Problem &problem) {
    auto data = readCustomerData(input);

    setCustomers(data, problem.customers);
    setDistances(data, problem.distances);
    setDurations(data, problem.durations);
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

    thrust::for_each(data.begin(), data.end(),
                     [&](const CustomerData &customer) {
                       customers.ids.push_back(thrust::get<0>(customer));
                       customers.demands.push_back(thrust::get<3>(customer));
                       customers.startTimes.push_back(thrust::get<4>(customer));
                       customers.endTimes.push_back(thrust::get<5>(customer));
                     });
  }

  /// Creates distance matrix.
  static void setDistances(const thrust::host_vector<CustomerData> &data,
                           thrust::device_vector<float> &distances) {
    // TODO move calculations on GPU
    typedef thrust::host_vector<CustomerData>::const_iterator Iterator;
    vrp::utils::repeated_range<Iterator> repeated(data.begin(), data.end(), data.size());
    vrp::utils::tiled_range<Iterator> tiled(data.begin(), data.end(), data.size());

    thrust::host_vector<float> hostDist(data.size() * data.size());

    thrust::transform(repeated.begin(), repeated.end(),
                      tiled.begin(),
                      hostDist.begin(),
                      DistanceCalculator());

    distances = hostDist;
  }

  /// Creates durations matrix.
  static void setDurations(const thrust::host_vector<CustomerData> &data,
                           thrust::device_vector<float> &durations) {
    durations.assign(data.size() * data.size(), 0);
  }
};

}
}

#endif //VRP_INPUT_SOLOMONREADER_HPP
