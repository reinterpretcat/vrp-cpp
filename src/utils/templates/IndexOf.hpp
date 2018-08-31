#ifndef VRP_UTILS_TEMPLATES_INDEXOF_HPP
#define VRP_UTILS_TEMPLATES_INDEXOF_HPP

#include <type_traits>

namespace vrp {
namespace utils {

template<typename...>
struct index_of;

template<class T, class... Rest>
struct index_of<T, T, Rest...> : std::integral_constant<std::size_t, 0u> {};

template<class T, class Other, class... Rest>
struct index_of<T, Other, Rest...>
  : std::integral_constant<std::size_t, 1 + index_of<T, Rest...>::value> {};

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_TEMPLATES_INDEXOF_HPP