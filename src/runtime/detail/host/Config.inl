#ifndef VRP_RUNTIME_DETAIL_HOST_CONFIG_HPP
#define VRP_RUNTIME_DETAIL_HOST_CONFIG_HPP

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

namespace vrp {
namespace runtime {

/// Alias for device execution policy.
#define EXEC_UNIT __host__

/// Specifies device execution policy.
struct exec_unit_policy : thrust::host_execution_policy<exec_unit_policy> {
} exec_unit;


}  // namespace runtime
}  // namespace vrp

#include "runtime/detail/host/Memory.inl"

namespace vrp {
namespace runtime {

/// Alias for vector.
template<typename T>
using vector = thrust::host_vector<T, detail::vector_allocator<T>>;

/// Alias for vector pointer.
template<typename T>
using vector_ptr = typename vector<T>::pointer;

}  // namespace runtime
}  // namespace vrp

#endif  // VRP_RUNTIME_DETAIL_HOST_CONFIG_HPP
