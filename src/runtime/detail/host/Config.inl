#ifndef VRP_RUNTIME_DETAIL_HOST_CONFIG_HPP
#define VRP_RUNTIME_DETAIL_HOST_CONFIG_HPP


#include <thrust/execution_policy.h>

namespace vrp {
namespace runtime {

/// Alias for device execution policy.
#define EXEC_UNIT __host__

/// Specifies device execution policy.
struct exec_unit_policy : thrust::host_execution_policy<exec_unit_policy> {};


}  // namespace runtime
}  // namespace vrp

#include "runtime/detail/host/Atomic.inl"
#include "runtime/detail/host/Containers.inl"
#include "runtime/detail/host/Memory.inl"
#include "runtime/detail/host/Operations.inl"


#endif  // VRP_RUNTIME_DETAIL_HOST_CONFIG_HPP
