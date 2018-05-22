#ifndef VRP_MODELS_CUSTOMERS_HPP
#define VRP_MODELS_CUSTOMERS_HPP

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represents a customer by "Struct of Array" idiom.
struct Customers final {
  /// Stores device pointers to data.
  struct Shadow final {
    thrust::device_ptr<const int> demands;
    thrust::device_ptr<const int> services;
    thrust::device_ptr<const int> starts;
    thrust::device_ptr<const int> ends;
  };

  /// Customer demand.
  thrust::device_vector<int> demands;

  /// Customer service times.
  thrust::device_vector<int> services;

  /// Customer time window start.
  thrust::device_vector<int> starts;

  /// Customer time window end.
  thrust::device_vector<int> ends;

  /// Reserves customers size.
  void reserve(std::size_t size) {
    demands.reserve(size);
    services.reserve(size);
    starts.reserve(size);
    ends.reserve(size);
  }

  /// Returns shadow object.
  Shadow getShadow() const { return {demands.data(), services.data(), starts.data(), ends.data()}; }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_CUSTOMERS_HPP
