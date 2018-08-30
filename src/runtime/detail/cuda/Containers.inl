#include <thrust/memory.h>

namespace vrp {
namespace runtime {

namespace detail {
/// Allocator used by device vector.
template<typename T>
struct vector_allocator : thrust::device_malloc_allocator<T> {
  typedef thrust::device_malloc_allocator<T> super_t;
  typedef typename super_t::pointer pointer;
  typedef typename super_t::size_type size_type;

  __host__ pointer allocate(size_type n) {
    // printf("vector_allocator::allocate() on device\n");

    return super_t::allocate(n);
  }

  // customize deallocate
  __host__ void deallocate(pointer p, size_type n) {
    // printf("vector_allocator::deallocate() on device\n");

    super_t::deallocate(p, n);
  }
};
}  // namespace detail

/// Alias for device vector.
template<typename T>
using vector = thrust::device_vector<T, detail::vector_allocator<T>>;

/// Alias for host vector pointer.
template<typename T>
using vector_ptr = typename vector<T>::pointer;

/// Alias for host vector const pointer.
template<typename T>
using vector_const_ptr = typename vector<T>::const_pointer;

}  // namespace runtime
}  // namespace vrp