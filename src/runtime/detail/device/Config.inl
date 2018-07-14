#ifndef VRP_RUNTIME_DETAIL_DEVICE_CONFIG_HPP
#define VRP_RUNTIME_DETAIL_DEVICE_CONFIG_HPP

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace vrp {
namespace runtime {

/// Alias for device execution policy.
#define EXEC_UNIT __device__

/// Specifies device execution policy.
struct exec_unit_policy : thrust::device_execution_policy<exec_unit_policy> {
};


}  // namespace runtime
}  // namespace vrp

#include "runtime/detail/device/Memory.inl"

namespace vrp {
namespace runtime {

/// Alias for vector.
template<typename T>
using vector = thrust::device_vector<T, detail::vector_allocator<T>>;

/// Alias for vector pointer.
template<typename T>
using vector_ptr = thrust::device_ptr<T>;

}  // namespace runtime
}  // namespace vrp

#endif  // VRP_RUNTIME_DETAIL_DEVICE_CONFIG_HPP
