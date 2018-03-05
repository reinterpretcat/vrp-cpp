#ifndef VRP_MODELS_SOLUTIONS_HPP
#define VRP_MODELS_SOLUTIONS_HPP

#include "models/Tasks.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace vrp {
namespace models {

/// Represents a feasible solution by "Struct of Array" idiom.
struct Solutions {

  /// Aggregated costs.
  thrust::device_vector<float> costs;

  /// Solution tasks.
  thrust::device_vector<thrust::device_ptr<Tasks>> tasks;
};

}
}

#endif //VRP_MODELS_SOLUTIONS_HPP
