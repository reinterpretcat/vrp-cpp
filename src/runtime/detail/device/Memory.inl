#include "runtime/detail/device/Config.inl"

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

namespace vrp {
namespace runtime {

// TODO cache memory

/// Allocates temporary buffer.
template<typename T>
thrust::pair<thrust::pointer<T, exec_unit_policy>, std::ptrdiff_t> get_temporary_buffer(
  exec_unit_policy,
  std::ptrdiff_t n) {
  printf("get_temporary_buffer(exec_unit_policy): calling device_malloc\n");

  // ask device_malloc for storage
  thrust::pointer<T, exec_unit_policy> result(thrust::device_malloc<T>(n).get());

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result, n);
}

/// Returns back temporary buffer.
template<typename Pointer>
void return_temporary_buffer(exec_unit_policy, Pointer p) {
  printf("return_temporary_buffer(exec_unit_policy): calling device_free\n");

  thrust::device_free(thrust::device_pointer_cast(p.get()));
}

/// Allocates buffer dynamically in device memory.
template<typename T>
EXEC_UNIT vrp::runtime::vector_ptr<T> allocate(size_t size) {
  return thrust::device_malloc<T>(size).get();
}

/// Allocates buffer to store single value in device memory.
template<typename T>
EXEC_UNIT vrp::runtime::vector_ptr<T> allocate(const T& value) {
  auto pValue = allocate<T>(1);
  *pValue = value;
  return pValue;
}

/// Deallocates dynamically allocated buffer.
template<typename T>
EXEC_UNIT void deallocate(vrp::runtime::vector_ptr<T> ptr) {
  thrust::device_free(ptr);
}

///  Deallocates dynamically allocated buffer and returns first item value.
template<typename T>
EXEC_UNIT inline T release(vrp::runtime::vector_ptr<T>& ptr) {
  T value = *ptr;
  deallocate(ptr);
  return value;
}

}  // namespace runtime
}  // namespace vrp
