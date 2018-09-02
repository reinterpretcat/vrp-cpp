#ifndef VRP_RUNTIME_DETAIL_OMP_CONFIG_HPP
#define VRP_RUNTIME_DETAIL_OMP_CONFIG_HPP

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace vrp {
namespace runtime {

#define ANY_EXEC_UNIT __host__ __device__

/// Alias for device execution policy.
#define EXEC_UNIT __device__

/// Specifies device execution policy.
struct exec_unit_policy : thrust::device_execution_policy<exec_unit_policy> {};


}  // namespace runtime
}  // namespace vrp

#include "runtime/detail/cpp/Atomic.inl"
#include "runtime/detail/cpp/Operations.inl"
#include "runtime/detail/cuda/Containers.inl"
#include "runtime/detail/cuda/Memory.inl"
#include "runtime/detail/host/Random.inl"


#endif  // VRP_RUNTIME_DETAIL_OMP_CONFIG_HPP
