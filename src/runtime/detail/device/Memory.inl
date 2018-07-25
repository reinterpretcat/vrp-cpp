#include "runtime/detail/device/Config.inl"

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

namespace vrp {
namespace runtime {

// TODO cache memory

/// Casts vector_ptr to raw pointer
template<typename T>
__host__ __device__ inline T* raw_pointer_cast(vector_ptr<T>& ptr) {
  return ptr.get();
}

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
EXEC_UNIT T* allocate_data(size_t size) {
  printf("allocate on device\n");
  return thrust::malloc<T>(thrust::device, size).get();
}

/// Allocates buffer to store single value in device memory.
template<typename T>
EXEC_UNIT T* allocate_value(const T& value) {
  auto pValue = allocate_data<T>(1);
  *pValue = value;
  return pValue;
}

/// Deallocates dynamically allocated buffer.
template<typename T>
EXEC_UNIT void deallocate(T* ptr) {
  printf("deallocate on device\n");
  thrust::free(thrust::device, ptr);
}

/// Allocates a new buffer in memory initializing with given value.
template<typename T>
ANY_EXEC_UNIT inline vector_ptr<T> allocate(const T& value) {
  vector_ptr<T> pValue = thrust::device_malloc<T>(1);
  *pValue = value;
  return pValue;
}

/// Releases buffer returning its value.
template<typename T>
ANY_EXEC_UNIT inline T release(vector_ptr<T>& buffer) {
  T value = *buffer;
  thrust::device_free(buffer);
  return value;
}

}  // namespace runtime
}  // namespace vrp
