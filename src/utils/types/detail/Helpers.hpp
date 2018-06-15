#ifndef VRP_UTILS_TYPES_DETAIL_HELPERS_HPP
#define VRP_UTILS_TYPES_DETAIL_HELPERS_HPP

#include <thrust/execution_policy.h>
#include <type_traits>

namespace vrp {
namespace utils {
namespace detail {

template<typename...>
struct one_of {
  static constexpr bool value = false;
};

template<typename T, typename S, typename... Ts>
struct one_of<T, S, Ts...> {
  static constexpr bool value = std::is_same<T, S>::value || one_of<T, Ts...>::value;
};

#include <type_traits>

template<typename...>
struct index_of;

template<class T, class... Rest>
struct index_of<T, T, Rest...> : std::integral_constant<std::size_t, 0u> {};

template<class T, class Other, class... Rest>
struct index_of<T, Other, Rest...>
  : std::integral_constant<std::size_t, 1 + index_of<T, Rest...>::value> {};


template<class... Ts>
struct VariantHelper;

template<class Union, class T, class... Ts>
struct VariantHelper<Union, T, Ts...> {
  __host__ __device__ inline static void destroy(std::size_t index, Union* data);
  __host__ __device__ inline static void move(std::size_t index, Union* oldValue, Union* newValue);
  __host__ __device__ inline static void copy(std::size_t index,
                                              const Union* oldValue,
                                              Union* new_v);
};

template<class Union>
struct VariantHelper<Union> {
  __host__ __device__ inline static void destroy(std::size_t index, Union* data) {}
  __host__ __device__ inline static void move(std::size_t index, Union* oldValue, Union* newValue) {
  }
  __host__ __device__ inline static void copy(std::size_t index,
                                              const Union* oldValue,
                                              Union* newValue) {}
};

}  // namespace detail
}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_TYPES_DETAIL_HELPERS_HPP