#ifndef VRP_RUNTIME_DEVICE_UNIQUE_HPP
#define VRP_RUNTIME_DEVICE_UNIQUE_HPP

#include "runtime/Config.hpp"

#include <thrust/swap.h>

namespace vrp {
namespace runtime {

/// Default deleter.
template<typename T>
struct default_delete final {
  EXEC_UNIT void operator()(T* ptr) const { delete ptr; }
};

/// Default deleter for arrays.
template<typename T>
struct default_delete<T[]> final {
  template<typename Arr>
  EXEC_UNIT typename std::enable_if<std::is_convertible<Arr (*)[], T (*)[]>::value>::type
  operator()(Arr* ptr) const {
    delete[] ptr;
  }
};

/// Smart pointer to use on device or host.
template<typename T, typename Deleter = default_delete<T>>
class unique_ptr final {
  T* data;
  Deleter deleter;

public:
  EXEC_UNIT unique_ptr() : data(nullptr), deleter(Deleter()) {}

  EXEC_UNIT explicit unique_ptr(T* data) : data(data), deleter(Deleter()) {}

  EXEC_UNIT explicit unique_ptr(std::nullptr_t) : data(nullptr), deleter(Deleter()) {}

  EXEC_UNIT unique_ptr(T* data, Deleter&& deleter) : data(data), deleter(deleter) {}

  EXEC_UNIT ~unique_ptr() { reset(); }


  EXEC_UNIT unique_ptr& operator=(std::nullptr_t) {
    reset();
    return *this;
  }

  EXEC_UNIT unique_ptr(unique_ptr<T, Deleter>&& moving) noexcept :
    data(std::move(moving.data)), deleter(std::move(moving.deleter)) {
    moving.data = nullptr;
  }

  EXEC_UNIT unique_ptr& operator=(unique_ptr<T, Deleter>&& moving) noexcept {
    moving.swap(*this);
    return *this;
  }

  EXEC_UNIT unique_ptr(unique_ptr const&) = delete;
  EXEC_UNIT unique_ptr& operator=(unique_ptr const&) = delete;

  EXEC_UNIT T* operator->() const { return data; }
  EXEC_UNIT T& operator*() const { return *data; }

  EXEC_UNIT T* get() const { return data; }

  EXEC_UNIT T* release() noexcept {
    T* result = nullptr;
    thrust::swap(result, data);
    return result;
  }

  EXEC_UNIT void swap(unique_ptr& src) noexcept {
    thrust::swap(data, src.data);
    thrust::swap(deleter, src.deleter);
  }

  EXEC_UNIT void reset() {
    if (data != nullptr) {
      T* tmp = release();
      deleter(tmp);
    }
  }
};

/// Creates device unique pointer.
template<typename T, typename Deleter = default_delete<T>, typename... Args>
EXEC_UNIT unique_ptr<T, Deleter> make_device_unique(Args&&... args) {
  return unique_ptr<T, Deleter>(new T(std::forward<Args>(args)...));
}

}  // namespace runtime
}  // namespace vrp

#endif  // VRP_RUNTIME_DEVICE_UNIQUE_HPP
