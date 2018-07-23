#ifndef VRP_RUNTIME_DEVICE_UNIQUE_HPP
#define VRP_RUNTIME_DEVICE_UNIQUE_HPP

#include "runtime/Config.hpp"

#include <thrust/swap.h>

namespace vrp {
namespace runtime {

/// Default deleter.
template<typename T>
struct default_delete final {
  EXEC_UNIT void operator()(T* ptr) const {
    printf("default_delete: deallocate value\n");
    deallocate(ptr);
  }
};

/// Default deleter for vector_ptr
template<typename T>
struct default_delete<vector_ptr<T>> final {
  size_t size;

  EXEC_UNIT void operator()(vector_ptr<T>* ptr) const {
    // TODO is memory allocated with make_unique_ptr_data cleaned fully?
    printf("default_delete: deallocate vector_ptr of size=%d\n", static_cast<int>(size));
    deallocate(ptr->get());
    deallocate(ptr);
  }
};

/// Default deleter for arrays.
template<typename T>
struct default_delete<T[]> final {
  template<typename Arr>
  EXEC_UNIT typename std::enable_if<std::is_convertible<Arr (*)[], T (*)[]>::value>::type
  operator()(Arr* ptr) const {
    // TODO investigate how this case is handled now.
    printf("default_delete: deallocate array\n");
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

/// Creates unique pointer to hold data of given size.
template<typename T, typename Deleter = default_delete<vector_ptr<T>>>
EXEC_UNIT unique_ptr<vector_ptr<T>, Deleter> make_unique_ptr_data(size_t size) {
  auto buffer = allocate_data<T>(size);
  auto vectorPtr = new vector_ptr<T>(buffer);
  return unique_ptr<vector_ptr<T>, Deleter>(vectorPtr, Deleter{size});
}

/// Creates unique pointer to hold single value.
template<typename T, typename Deleter = default_delete<T>, typename... Args>
EXEC_UNIT unique_ptr<T, Deleter> make_unique_ptr_value(Args&&... args) {
  return unique_ptr<T, Deleter>(allocate_value<T>(T(std::forward<Args>(args)...)));
}

}  // namespace runtime
}  // namespace vrp

#endif  // VRP_RUNTIME_DEVICE_UNIQUE_HPP
