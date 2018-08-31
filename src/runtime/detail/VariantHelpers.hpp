#ifndef VRP_RUNTIME_DETAIL_VARIANTHELPERS_HPP
#define VRP_RUNTIME_DETAIL_VARIANTHELPERS_HPP

#include "utils/templates/IndexOf.hpp"
#include "utils/templates/OneOf.hpp"

#include <thrust/execution_policy.h>

namespace vrp {
namespace runtime {
namespace detail {

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