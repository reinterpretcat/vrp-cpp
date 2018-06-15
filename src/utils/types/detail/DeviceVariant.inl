#include "utils/types/detail/Helpers.hpp"

#include <thrust/system_error.h>

namespace vrp {
namespace utils {
namespace detail {

template<class Union, class T, class... Ts>
void VariantHelper<Union, T, Ts...>::destroy(std::size_t index, Union* data) {
  if (index == 0u) {
    reinterpret_cast<T*>(data)->~T();
  } else {
    --index;
    VariantHelper<Union, Ts...>::destroy(index, data);
  }
}

template<class Union, class T, class... Ts>
void VariantHelper<Union, T, Ts...>::move(std::size_t index, Union* oldValue, Union* newValue) {
  if (index == 0u) {
    new (newValue) T(std::move(*reinterpret_cast<T*>(oldValue)));
  } else {
    --index;
    VariantHelper<Union, Ts...>::move(index, oldValue, newValue);
  }
}

template<class Union, class T, class... Ts>
void VariantHelper<Union, T, Ts...>::copy(std::size_t index,
                                          const Union* oldValue,
                                          Union* newValue) {
  if (index == 0u) {
    new (newValue) T(*reinterpret_cast<const T*>(oldValue));
  } else {
    --index;
    VariantHelper<Union, Ts...>::copy(index, oldValue, newValue);
  }
}

}  // namespace detail

template<class... Ts>
device_variant<Ts...>::~device_variant() {
  if (valid()) Helper::destroy(index - 1u, &data);
}


template<class... Ts>
device_variant<Ts...>::device_variant(const device_variant<Ts...>& other) : index{other.index} {
  if (valid()) Helper::copy(index - 1u, &other.data, &data);
}

template<class... Ts>
device_variant<Ts...>::device_variant(device_variant<Ts...>&& other) : index{other.index} {
  if (valid()) Helper::move(index - 1u, &other.data, &data);
}


template<class... Ts>
device_variant<Ts...>& device_variant<Ts...>::operator=(const device_variant<Ts...>& other) {
  if (&other != this) {
    if (valid()) Helper::destroy(index - 1u, &data);

    index = other.index;

    if (valid()) Helper::copy(index - 1u, &other.data, &data);
  }

  return *this;
}

template<class... Ts>
device_variant<Ts...>& device_variant<Ts...>::operator=(device_variant<Ts...>&& other) {
  if (&other != this) {
    if (valid()) Helper::destroy(index - 1u, &data);

    index = other.index;

    if (valid()) Helper::move(index - 1u, &other.data, &data);
  }

  return *this;
}

template<class... Ts>
template<class T>
bool device_variant<Ts...>::is() const {
  return index == detail::index_of<T, void, Ts...>::value;
}

template<class... Ts>
bool device_variant<Ts...>::valid() const {
  return index != 0u;
}


template<class... Ts>
template<class T, class... Args, class>
void device_variant<Ts...>::set(Args&&... args) {
  if (valid()) Helper::destroy(index - 1u, &data);

  new (&data) T(std::forward<Args>(args)...);
  index = detail::index_of<T, void, Ts...>::value;
}

// TODO check in getters
// index==detail::index_of<T, void, Ts...>::value;

template<class... Ts>
template<class T, class>
const T& device_variant<Ts...>::get() const {
  assert(valid() && "device_variant is not initialized.");
  return *reinterpret_cast<const T*>(&data);
}

template<class... Ts>
template<class T, class>
T& device_variant<Ts...>::get() {
  assert(valid() && "device_variant is not initialized.");
  return *reinterpret_cast<T*>(&data);
}

template<class... Ts>
void device_variant<Ts...>::reset() {
  if (valid()) Helper::destroy(index - 1u, &data);

  index = 0u;
}

}  // namespace utils
}  // namespace vrp
