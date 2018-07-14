#ifndef VRP_MODELS_CUSTOMERS_HPP
#define VRP_MODELS_CUSTOMERS_HPP

#include "runtime/Config.hpp"

namespace vrp {
namespace models {

/// Represents a customer by "Struct of Array" idiom.
struct Customers final {
  /// Stores device pointers to data.
  struct Shadow final {
    vrp::runtime::vector_ptr<const int> demands;
    vrp::runtime::vector_ptr<const int> services;
    vrp::runtime::vector_ptr<const int> starts;
    vrp::runtime::vector_ptr<const int> ends;
  };

  /// Customer demand.
  vrp::runtime::vector<int> demands;

  /// Customer service times.
  vrp::runtime::vector<int> services;

  /// Customer time window start.
  vrp::runtime::vector<int> starts;

  /// Customer time window end.
  vrp::runtime::vector<int> ends;

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
