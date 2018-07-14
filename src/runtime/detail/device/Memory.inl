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
  std::cout << "get_temporary_buffer(exec_unit_policy): calling device_malloc" << std::endl;

  // ask device_malloc for storage
  thrust::pointer<T, exec_unit_policy> result(thrust::device_malloc<T>(n).get());

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result, n);
}

template<typename Pointer>
void return_temporary_buffer(exec_unit_policy, Pointer p) {
  std::cout << "return_temporary_buffer(exec_unit_policy): calling device_free" << std::endl;

  thrust::device_free(thrust::device_pointer_cast(p.get()));
}

namespace detail {
/// Allocator used by device vector.
template<typename T>
struct vector_allocator : thrust::device_malloc_allocator<T> {
  typedef thrust::device_malloc_allocator<T> super_t;
  typedef typename super_t::pointer pointer;
  typedef typename super_t::size_type size_type;

  pointer allocate(size_type n) {
    std::cout << "vector_allocator::allocate() on device" << std::endl;

    return super_t::allocate(n);
  }

  // customize deallocate
  void deallocate(pointer p, size_type n) {
    std::cout << "vector_allocator::deallocate() on device:" << std::endl;

    super_t::deallocate(p, n);
  }
};
}  // namespace detail

}  // namespace runtime
}  // namespace vrp
