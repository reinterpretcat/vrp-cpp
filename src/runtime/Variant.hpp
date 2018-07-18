#ifndef VRP_RUNTIME_DEVICEVARIANT_HPP
#define VRP_RUNTIME_DEVICEVARIANT_HPP

#include "runtime/detail/VariantHelpers.hpp"

#include <cassert>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility>

namespace vrp {
namespace runtime {

/// Implements variant type which works on device.
template<class... Ts>
class variant {
public:
  static_assert(sizeof...(Ts) > 1, "Variant should have at least 2 different types.");

  __host__ __device__ variant() = default;
  __host__ __device__ ~variant();

  __host__ __device__ variant(const variant<Ts...>& other);
  __host__ __device__ variant(variant<Ts...>&& other);


  __host__ __device__ variant<Ts...>& operator=(const variant<Ts...>& other);
  __host__ __device__ variant<Ts...>& operator=(variant<Ts...>&& other);

  template<class T>
  __host__ __device__ bool is() const;

  __host__ __device__ bool valid() const;


  template<class T,
           class... Args,
           class = typename std::enable_if<detail::one_of<T, Ts...>::value>::type>
  __host__ __device__ void set(Args&&... args);

  template<class T, class = typename std::enable_if<detail::one_of<T, Ts...>::value>::type>
  __host__ __device__ const T& get() const;

  template<class T, class = typename std::enable_if<detail::one_of<T, Ts...>::value>::type>
  __host__ __device__ T& get();

  __host__ __device__ void reset();

private:
  using Data = typename std::aligned_union<0, Ts...>::type;
  using Helper = detail::VariantHelper<Data, Ts...>;

  std::size_t index{};
  Data data;
};

}  // namespace runtime
}  // namespace vrp

#include "runtime/detail/Variant.inl"

#endif  // VRP_RUNTIME_DEVICEVARIANT_HPP
