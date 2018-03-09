#ifndef VRP_MODELS_CUSTOMERS_HPP
#define VRP_MODELS_CUSTOMERS_HPP

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represents a customer by "Struct of Array" idiom.
struct Customers {

  /// Customer id.
  thrust::device_vector<int> ids;

  /// Customer demand.
  thrust::device_vector<int> demands;

  /// Customer time window start.
  thrust::device_vector<int> startTimes;

  /// Customer time window end.
  thrust::device_vector<int> endTimes;

  /// Reserves customers size.
  void reserve(std::size_t size) {
    ids.reserve(size);
    demands.reserve(size);
    startTimes.reserve(size);
    endTimes.reserve(size);
  }
};

}
}

#endif //VRP_MODELS_CUSTOMERS_HPP
