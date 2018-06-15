#ifndef VRP_MODELS_PLAN_HPP
#define VRP_MODELS_PLAN_HPP

#include <thrust/pair.h>

namespace vrp {
namespace models {

/// Represents an entity which stores information about
/// assignment plan for specific customer:
/// - whether it is assigned or not;
/// - whether it is part of specific convolution.
struct Plan final {
  constexpr static int NoConvolution = -1;

  __host__ __device__ Plan() : data(thrust::make_pair(false, NoConvolution)) {}

  __host__ __device__ explicit Plan(const thrust::pair<bool, int> data) : data(data) {}

  /// Returns default plan which has no assignment.
  __host__ __device__ static Plan empty() { return Plan{}; }

  /// Marks plan as assigned with convolution.
  __host__ __device__ static Plan assign(int index) { return Plan{thrust::make_pair(true, index)}; }

  /// Creates plan as assigned with customer.
  __host__ __device__ static Plan assign() { return Plan{thrust::make_pair(true, NoConvolution)}; }

  /// Creates plan as reserved with convolution.
  __host__ __device__ static Plan reserve(int index) {
    return Plan{thrust::make_pair(false, index)};
  }

  /// Returns false if plan is not set.
  __host__ __device__ bool isAssigned() const { return data.first; }

  /// Returns false if convolution is not used.
  __host__ __device__ bool hasConvolution() const { return data.second != NoConvolution; };

  /// Returns convolution index.
  __host__ __device__ int convolution() const { return data.second; }

  __host__ __device__ bool compare(const Plan& other) const { return data == other.data; }

private:
  thrust::pair<bool, int> data;
};

inline bool operator==(const Plan& lhs, const Plan& rhs) { return lhs.compare(rhs); }
inline bool operator!=(const Plan& lhs, const Plan& rhs) { return !lhs.compare(rhs); }

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_PLAN_HPP
