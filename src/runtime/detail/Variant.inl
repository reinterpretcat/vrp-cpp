#include "runtime/detail/VariantHelpers.hpp"

#include <thrust/system_error.h>

namespace vrp {
namespace runtime {
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
variant<Ts...>::~variant() {
  if (valid()) Helper::destroy(index - 1u, &data);
}


template<class... Ts>
variant<Ts...>::variant(const variant<Ts...>& other) : index{other.index} {
  if (valid()) Helper::copy(index - 1u, &other.data, &data);
}

template<class... Ts>
variant<Ts...>::variant(variant<Ts...>&& other) : index{other.index} {
  if (valid()) Helper::move(index - 1u, &other.data, &data);
}


template<class... Ts>
variant<Ts...>& variant<Ts...>::operator=(const variant<Ts...>& other) {
  if (&other != this) {
    if (valid()) Helper::destroy(index - 1u, &data);

    index = other.index;

    if (valid()) Helper::copy(index - 1u, &other.data, &data);
  }

  return *this;
}

template<class... Ts>
variant<Ts...>& variant<Ts...>::operator=(variant<Ts...>&& other) {
  if (&other != this) {
    if (valid()) Helper::destroy(index - 1u, &data);

    index = other.index;

    if (valid()) Helper::move(index - 1u, &other.data, &data);
  }

  return *this;
}

template<class... Ts>
template<class T>
bool variant<Ts...>::is() const {
  return index == utils::index_of<T, void, Ts...>::value;
}

template<class... Ts>
bool variant<Ts...>::valid() const {
  return index != 0u;
}


template<class... Ts>
template<class T, class... Args, class>
void variant<Ts...>::set(Args&&... args) {
  if (valid()) Helper::destroy(index - 1u, &data);

  new (&data) T(std::forward<Args>(args)...);
  index = utils::index_of<T, void, Ts...>::value;
}

// TODO check in getters
// index==utils::index_of<T, void, Ts...>::value;

template<class... Ts>
template<class T, class>
const T& variant<Ts...>::get() const {
  assert(valid() && "variant is not initialized.");
  assert(is<T>() && "wrong type requested.");
  return *reinterpret_cast<const T*>(&data);
}

template<class... Ts>
template<class T, class>
T& variant<Ts...>::get() {
  assert(valid() && "variant is not initialized.");
  assert(is<T>() && "wrong type requested.");
  return *reinterpret_cast<T*>(&data);
}

template<class... Ts>
void variant<Ts...>::reset() {
  if (valid()) Helper::destroy(index - 1u, &data);

  index = 0u;
}

}  // namespace runtime
}  // namespace vrp
