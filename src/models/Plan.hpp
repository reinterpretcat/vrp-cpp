#ifndef VRP_MODELS_PLAN_HPP
#define VRP_MODELS_PLAN_HPP

#include "runtime/Config.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace models {

/// Represents an entity which stores information about
/// assignment plan for specific customer:
/// - whether it is assigned or not;
/// - whether it is part of specific convolution.
struct Plan final {
  constexpr static int NoConvolution = -1;

  ANY_EXEC_UNIT Plan() : data(thrust::make_pair(false, NoConvolution)) {}

  ANY_EXEC_UNIT explicit Plan(const thrust::pair<bool, int> data) : data(data) {}

  /// Returns default plan which has no assignment.
  ANY_EXEC_UNIT static Plan empty() { return Plan{}; }

  /// Marks plan as assigned with convolution.
  ANY_EXEC_UNIT static Plan assign(int index) { return Plan{thrust::make_pair(true, index)}; }

  /// Creates plan as assigned with customer.
  ANY_EXEC_UNIT static Plan assign() { return Plan{thrust::make_pair(true, NoConvolution)}; }

  /// Creates plan as reserved with convolution.
  ANY_EXEC_UNIT static Plan reserve(int index) { return Plan{thrust::make_pair(false, index)}; }

  /// Returns false if plan is not set.
  ANY_EXEC_UNIT bool isAssigned() const { return data.first; }

  /// Returns false if convolution is not used.
  ANY_EXEC_UNIT bool hasConvolution() const { return data.second != NoConvolution; };

  /// Returns convolution index.
  ANY_EXEC_UNIT int convolution() const { return data.second; }

  ANY_EXEC_UNIT bool compare(const Plan& other) const { return data == other.data; }

private:
  thrust::pair<bool, int> data;
};

inline bool operator==(const Plan& lhs, const Plan& rhs) { return lhs.compare(rhs); }
inline bool operator!=(const Plan& lhs, const Plan& rhs) { return !lhs.compare(rhs); }

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_PLAN_HPP
