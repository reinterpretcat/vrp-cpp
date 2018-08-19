#include "runtime/detail/host/Config.inl"

#include <thrust/memory.h>

namespace vrp {
namespace runtime {

// TODO cache memory
/// Casts vector_ptr to raw pointer
template<typename T>
ANY_EXEC_UNIT inline T* raw_pointer_cast(vector_ptr<T>& ptr) {
  return static_cast<T*>(ptr);
}

template<typename T>
thrust::pair<thrust::pointer<T, exec_unit_policy>, std::ptrdiff_t> get_temporary_buffer(
  exec_unit_policy exec_unit,
  std::ptrdiff_t n) {
  //std::cout << "get_temporary_buffer(exec_unit_policy): host" << std::endl;

  // ask device_malloc for storage
  thrust::pointer<T, exec_unit_policy> result(thrust::malloc<T>(exec_unit, n).get());

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result, n);
}

template<typename Pointer>
void return_temporary_buffer(exec_unit_policy exec_unit, Pointer p) {
  //std::cout << "return_temporary_buffer(exec_unit_policy): host" << std::endl;

  thrust::free(exec_unit, p.get());
}

/// Allocates buffer dynamically in host memory.
template<typename T>
EXEC_UNIT T* allocate_data(size_t size) {
  //printf("allocate on host\n");
  return thrust::malloc<T>(thrust::host, size).get();
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
  thrust::free(exec_unit_policy{}, ptr);
}


/// Allocates a new buffer in memory initializing with given value.
template<typename T>
ANY_EXEC_UNIT inline vector_ptr<T> allocate(const T& value) {
  vector_ptr<T> pValue = vector_ptr<T>(thrust::malloc<T>(exec_unit_policy{}, 1).get());
  *pValue = value;
  return pValue;
}

/// Releases buffer returning its value.
template<typename T>
ANY_EXEC_UNIT inline T release(vector_ptr<T> buffer) {
  T value = *buffer;
  thrust::free(exec_unit_policy{}, buffer);
  return value;
}

}  // namespace runtime
}  // namespace vrp
