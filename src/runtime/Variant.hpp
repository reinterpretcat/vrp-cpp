#ifndef VRP_RUNTIME_DEVICEVARIANT_HPP
#define VRP_RUNTIME_DEVICEVARIANT_HPP

#include "runtime/Config.hpp"
#include "runtime/detail/VariantHelpers.hpp"

#include <cassert>
#include <thrust/execution_policy.h>
#include <utility>

namespace vrp {
namespace runtime {

/// Implements variant type which works on device.
template<class... Ts>
class variant {
public:
  static_assert(sizeof...(Ts) > 1, "Variant should have at least 2 different types.");

  ANY_EXEC_UNIT variant() = default;
  ANY_EXEC_UNIT ~variant();

  ANY_EXEC_UNIT variant(const variant<Ts...>& other);
  ANY_EXEC_UNIT variant(variant<Ts...>&& other);


  ANY_EXEC_UNIT variant<Ts...>& operator=(const variant<Ts...>& other);
  ANY_EXEC_UNIT variant<Ts...>& operator=(variant<Ts...>&& other);

  template<class T>
  ANY_EXEC_UNIT bool is() const;

  ANY_EXEC_UNIT bool valid() const;


  template<class T,
           class... Args,
           class = typename std::enable_if<utils::one_of<T, Ts...>::value>::type>
  ANY_EXEC_UNIT void set(Args&&... args);

  template<class T, class = typename std::enable_if<utils::one_of<T, Ts...>::value>::type>
  ANY_EXEC_UNIT const T& get() const;

  template<class T, class = typename std::enable_if<utils::one_of<T, Ts...>::value>::type>
  ANY_EXEC_UNIT T& get();

  ANY_EXEC_UNIT void reset();

  template<class T, class = typename std::enable_if<utils::one_of<T, Ts...>::value>::type>
  ANY_EXEC_UNIT static variant<Ts...> create(T value) {
    variant<Ts...> v;
    v.set<T>(value);
    return v;
  }

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
