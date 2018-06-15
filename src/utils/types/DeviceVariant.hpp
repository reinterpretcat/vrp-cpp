#ifndef VRP_UTILS_TYPES_DEVICEVARIANT_HPP
#define VRP_UTILS_TYPES_DEVICEVARIANT_HPP

#include "utils/types/detail/Helpers.hpp"

#include <cassert>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility>

namespace vrp {
namespace utils {

/// Implements variant type which works on device.
template<class... Ts>
class device_variant {
public:
  static_assert(sizeof...(Ts) > 1, "Variant should have at least 2 different types.");

  __host__ __device__ device_variant() = default;
  __host__ __device__ ~device_variant();

  __host__ __device__ device_variant(const device_variant<Ts...>& other);
  __host__ __device__ device_variant(device_variant<Ts...>&& other);


  __host__ __device__ device_variant<Ts...>& operator=(const device_variant<Ts...>& other);
  __host__ __device__ device_variant<Ts...>& operator=(device_variant<Ts...>&& other);

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

}  // namespace utils
}  // namespace vrp

#include "utils/types/detail/DeviceVariant.inl"

#endif  // VRP_UTILS_TYPES_DEVICEVARIANT_HPP
