#include <thrust/host_vector.h>

namespace vrp {
namespace runtime {

namespace detail {
/// Allocator used by host vector.
template<typename T>
struct vector_allocator : std::allocator<T> {
  typedef std::allocator<T> super_t;
  typedef typename super_t::pointer pointer;
  typedef typename super_t::size_type size_type;

  pointer allocate(size_type n) {
    std::cout << "vector_allocator::allocate() host" << std::endl;

    return super_t::allocate(n);
  }

  // customize deallocate
  void deallocate(pointer p, size_type n) {
    std::cout << "vector_allocator::deallocate() host" << std::endl;

    super_t::deallocate(p, n);
  }
};

}  // namespace detail

/// Alias for host vector.
template<typename T>
using vector = thrust::host_vector<T, detail::vector_allocator<T>>;

/// Alias for host vector pointer.
template<typename T>
using vector_ptr = typename vector<T>::pointer;

}  // namespace runtime
}  // namespace vrp