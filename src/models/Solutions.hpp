#ifndef VRP_MODELS_SOLUTIONS_HPP
#define VRP_MODELS_SOLUTIONS_HPP

#include "models/Tasks.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace vrp {
namespace models {

/// Represents a feasible solution by "Struct of Array" idiom.
struct Solutions {
  /// Defines tour (aka route) type.
  using Tours = thrust::device_ptr<Tasks>;

  /// Aggregated costs.
  thrust::device_vector<float> costs;

  /// Solution tours.
  thrust::device_vector<Tours> tasks;
};

}
}

#endif //VRP_MODELS_SOLUTIONS_HPP
