#include "runtime/detail/device/Config.inl"

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/memory.h>

namespace vrp {
namespace runtime {

// TODO cache memory

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

template<typename Pointer>
void return_temporary_buffer(exec_unit_policy, Pointer p) {
  printf("return_temporary_buffer(exec_unit_policy): calling device_free\n");

  thrust::device_free(thrust::device_pointer_cast(p.get()));
}

namespace detail {
/// Allocator used by device vector.
template<typename T>
struct vector_allocator : thrust::device_malloc_allocator<T> {
  typedef thrust::device_malloc_allocator<T> super_t;
  typedef typename super_t::pointer pointer;
  typedef typename super_t::size_type size_type;

  __host__ pointer allocate(size_type n) {
    printf("vector_allocator::allocate() on device\n");

    return super_t::allocate(n);
  }

  // customize deallocate
  __host__ void deallocate(pointer p, size_type n) {
    printf("vector_allocator::deallocate() on device\n");

    super_t::deallocate(p, n);
  }
};
}  // namespace detail

}  // namespace runtime
}  // namespace vrp
