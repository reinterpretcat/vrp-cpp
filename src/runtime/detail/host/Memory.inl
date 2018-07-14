#include "runtime/detail/host/Config.inl"

#include <thrust/memory.h>

namespace vrp {
namespace runtime {

// TODO cache memory

template<typename T>
thrust::pair<thrust::pointer<T, exec_unit_policy>, std::ptrdiff_t> get_temporary_buffer(
  exec_unit_policy exec_unit,
  std::ptrdiff_t n) {
  std::cout << "get_temporary_buffer(exec_unit_policy): host" << std::endl;

  // ask device_malloc for storage
  thrust::pointer<T, exec_unit_policy> result(thrust::malloc<T>(exec_unit, n).get());

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result, n);
}

template<typename Pointer>
void return_temporary_buffer(exec_unit_policy exec_unit, Pointer p) {
  std::cout << "return_temporary_buffer(exec_unit_policy): host" << std::endl;

  thrust::free(exec_unit, p.get());
}

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
}  // namespace runtime
}  // namespace vrp
