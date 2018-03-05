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
};

}
}

#endif //VRP_MODELS_CUSTOMERS_HPP
