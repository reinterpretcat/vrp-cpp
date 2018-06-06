#ifndef VRP_UTILS_MEMORY_DEVICE_UNIQUE_HPP
#define VRP_UTILS_MEMORY_DEVICE_UNIQUE_HPP

#include <thrust/execution_policy.h>
#include <thrust/swap.h>

namespace vrp {
namespace utils {

/// Default deleter.
template<typename T>
struct default_delete final {
  __device__ void operator()(T* ptr) const { delete ptr; }
};

/// Default deleter for arrays.
template<typename T>
struct default_delete<T[]> final {
  template<typename Arr>
  __device__ typename std::enable_if<std::is_convertible<Arr (*)[], T (*)[]>::value>::type
  operator()(Arr* ptr) const {
    delete[] ptr;
  }
};

/// Smart pointer to use on device.
template<typename T, typename Deleter = default_delete<T>>
class device_unique_ptr final {
  T* data;
  Deleter deleter;

public:
  __device__ device_unique_ptr() : data(nullptr), deleter(Deleter()) {}

  __device__ explicit device_unique_ptr(T* data) : data(data), deleter(Deleter()) {}

  __device__ explicit device_unique_ptr(std::nullptr_t) : data(nullptr), deleter(Deleter()) {}

  __device__ device_unique_ptr(T* data, Deleter&& deleter) : data(data), deleter(deleter) {}

  __device__ ~device_unique_ptr() { reset(); }


  __device__ device_unique_ptr& operator=(std::nullptr_t) {
    reset();
    return *this;
  }

  __device__ device_unique_ptr(device_unique_ptr<T, Deleter>&& moving) noexcept :
    data(std::move(moving.data)), deleter(std::move(moving.deleter)) {}

  __device__ device_unique_ptr& operator=(device_unique_ptr<T, Deleter>&& moving) noexcept {
    moving.swap(*this);
    return *this;
  }

  __device__ device_unique_ptr(device_unique_ptr const&) = delete;
  __device__ device_unique_ptr& operator=(device_unique_ptr const&) = delete;

  __device__ T* operator->() const { return data; }
  __device__ T& operator*() const { return *data; }

  __device__ T* get() const { return data; }

  __device__ T* release() noexcept {
    T* result = nullptr;
    thrust::swap(result, data);
    return result;
  }

  __device__ void swap(device_unique_ptr& src) noexcept {
    thrust::swap(data, src.data);
    thrust::swap(deleter, src.deleter);
  }

  __device__ void reset() {
    if (data != nullptr) {
      T * tmp = release();
      deleter(tmp);
    }
  }
};

/// Creates device unique pointer.
template<typename T, typename Deleter = default_delete<T>, typename... Args>
__device__ device_unique_ptr<T, Deleter> make_device_unique(Args&&... args) {
  return device_unique_ptr<T, Deleter>(new T(std::forward<Args>(args)...));
}

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_MEMORY_DEVICE_UNIQUE_HPP
