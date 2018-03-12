#ifndef VRP_MODELS_PROBLEM_HPP
#define VRP_MODELS_PROBLEM_HPP

#include "models/Customers.hpp"
#include "models/Resources.hpp"
#include "models/RoutingMatrix.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace vrp {
namespace models {

/// Represents a Vehicle Routing Problem.
struct Problem final {
  /// Stores device pointers to data.
  struct Shadow final {
    int size;
    Customers::Shadow customers;
    Resources::Shadow resources;
    RoutingMatrix::Shadow routing;
  };

  /// Customers to serve.
  Customers customers;

  /// Available resources.
  Resources resources;

  /// Routing matrix data.
  RoutingMatrix routing;

  /// Returns problem size.
  int size() const {
    return static_cast<int>(customers.demands.size());
  }

  /// Returns shadow object.
  Shadow getShadow() const {
    return {size(),
            customers.getShadow(),
            resources.getShadow(),
            routing.getShadow()};
  }
};

}
}

#endif //VRP_MODELS_PROBLEM_HPP
