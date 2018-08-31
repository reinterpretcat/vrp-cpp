#ifndef VRP_UTILS_TEMPLATES_ONEOF_HPP
#define VRP_UTILS_TEMPLATES_ONEOF_HPP

#include <type_traits>

namespace vrp {
namespace utils {

template<typename...>
struct one_of {
  static constexpr bool value = false;
};

template<typename T, typename S, typename... Ts>
struct one_of<T, S, Ts...> {
  static constexpr bool value = std::is_same<T, S>::value || one_of<T, Ts...>::value;
};

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_TEMPLATES_ONEOF_HPP