#ifndef VRP_RUNTIME_DETAIL_VARIANTHELPERS_HPP
#define VRP_RUNTIME_DETAIL_VARIANTHELPERS_HPP

#include <thrust/execution_policy.h>
#include <type_traits>

namespace vrp {
namespace runtime {
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
  ANY_EXEC_UNIT inline static void destroy(std::size_t index, Union* data);
  ANY_EXEC_UNIT inline static void move(std::size_t index, Union* oldValue, Union* newValue);
  ANY_EXEC_UNIT inline static void copy(std::size_t index, const Union* oldValue, Union* new_v);
};

template<class Union>
struct VariantHelper<Union> {
  ANY_EXEC_UNIT inline static void destroy(std::size_t index, Union* data) {}
  ANY_EXEC_UNIT inline static void move(std::size_t index, Union* oldValue, Union* newValue) {}
  ANY_EXEC_UNIT inline static void copy(std::size_t index, const Union* oldValue, Union* newValue) {
  }
};

}  // namespace detail
}  // namespace runtime
}  // namespace vrp

#endif  // VRP_RUNTIME_DETAIL_VARIANTHELPERS_HPP