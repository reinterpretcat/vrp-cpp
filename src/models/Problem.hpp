#ifndef VRP_MODELS_PROBLEM_HPP
#define VRP_MODELS_PROBLEM_HPP

#include "models/Customers.hpp"
#include "models/Resources.hpp"

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represents a Vehicle Routing Problem.
struct Problem {

  /// Customers to serve.
  Customers customers;

  /// Matrix of distances.
  thrust::device_vector<float> distances;

  /// Matrix of durations.
  thrust::device_vector<float> durations;

  /// Available resources.
  Resources resources;

  /// Returns problem size.
  int size() const {
    return static_cast<int>(customers.ids.size());
  }
};

}
}

#endif //VRP_MODELS_PROBLEM_HPP
