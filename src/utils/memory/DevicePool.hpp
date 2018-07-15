#ifndef VRP_UTILS_MEMORY_DEVICEPOOL_HPP
#define VRP_UTILS_MEMORY_DEVICEPOOL_HPP

#include "algorithms/convolutions/Models.hpp"
#include "models/Convolution.hpp"
#include "runtime/DeviceUnique.hpp"

#include <functional>
#include <memory>
#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

namespace vrp {
namespace utils {

/// Implements object pool for used types which works on device.
/// The main goal of this implementation is to allow dynamically
/// reuse memory allocated on device and keep interfaces clean.
/// Some notes:
/// - as RTTI is not available on device, only specify used types.
/// - destroy function must be called to release pool's memory.
/// - it is not intended to be used in concurrent context, so there
///   is no synchronization.
class DevicePool final {
public:
  using Deleter = std::function<void(void*)>;
  using Pointer = thrust::device_ptr<DevicePool>;
  using UniquePointer = std::unique_ptr<Pointer, Deleter>;

  /// Provides pool functionality for specific element type.
  template<typename T>
  class TypedPool final {
  public:
    /// Pool deleter which assumes pool lives longer.
    struct pool_vector_deleter final {
      __device__ explicit pool_vector_deleter(TypedPool* pool) : pool(pool) {}

      __device__ void operator()(thrust::device_ptr<T>* ptr) { pool->release(*ptr); }

    private:
      TypedPool* pool;
    };

    using DataType = thrust::device_ptr<T>;
    using ReturnType = vrp::runtime::device_unique_ptr<DataType, pool_vector_deleter>;

    friend DevicePool;

    TypedPool(size_t capacity, size_t elements) : top(0u), capacity(capacity), elements(elements) {
      data = thrust::device_new<DataType>(capacity);
      for (; top < capacity; ++top) {
        data[top] = thrust::device_new<T>(elements);
      }
    }

    __device__ ReturnType operator()() {
      return vrp::runtime::device_unique_ptr<DataType, pool_vector_deleter>(
        new DataType(data[--top]), pool_vector_deleter(this));
    }

  private:
    __device__ void release(DataType ptr) { data[top++] = ptr; }

    size_t top, capacity, elements;
    thrust::device_ptr<DataType> data;
  };

  static UniquePointer create(size_t count, size_t capacity, size_t elements) {
    auto buffer = thrust::device_malloc(sizeof(DevicePool) * count);
    auto pool = new Pointer(thrust::device_new(buffer, DevicePool(capacity, elements), count));
    return UniquePointer(pool, [=](void* ptr) {
      // TODO clean pool
      // auto* poolPtr = static_cast<Pointer*>(ptr);
    });
  }

  /// Returns pointer to int array.
  __device__ typename TypedPool<int>::ReturnType ints(size_t size) {
    assert(size <= elements);
    return intPool.get()->operator()();
  }

  /// Returns pointer to float array.
  __device__ typename TypedPool<float>::ReturnType floats(size_t size) {
    assert(size <= elements);
    return floatPool.get()->operator()();
  }

  /// Returns pointer to bool array.
  __device__ typename TypedPool<bool>::ReturnType bools(size_t size) {
    assert(size <= elements);
    return boolPool.get()->operator()();
  }

  /// Returns pointer to tuple<bool,int> array.
  __device__ typename TypedPool<thrust::tuple<bool, int>>::ReturnType boolInts(size_t size) {
    assert(size <= elements);
    return boolIntPool.get()->operator()();
  }

  /// Returns pointer to convolution array.
  __device__ typename TypedPool<vrp::models::Convolution>::ReturnType convolutions(size_t size) {
    assert(size <= elements);
    return convPool.get()->operator()();
  }

  /// Returns pointer to joint pair array.
  __device__ typename TypedPool<vrp::models::JointPair>::ReturnType jointPairs(size_t size) {
    assert(size <= elements);
    return jointPairPool.get()->operator()();
  }

private:
  DevicePool(size_t capacity, size_t elements) : capacity(capacity), elements(elements) {
    intPool = createPool<int>(capacity, elements);
    floatPool = createPool<float>(capacity, elements);
    boolPool = createPool<bool>(capacity, elements);
    boolIntPool = createPool<thrust::tuple<bool, int>>(capacity, elements);
    // TODO investigate theoretical max sizes
    convPool = createPool<vrp::models::Convolution>(capacity, elements / 2);
    jointPairPool = createPool<vrp::models::JointPair>(capacity, elements);
  }

  template<typename T>
  static thrust::device_ptr<TypedPool<T>> createPool(size_t capacity, size_t elements) {
    auto buffer = thrust::device_malloc(sizeof(TypedPool<T>));
    return thrust::device_new(buffer, TypedPool<T>(capacity, elements), 1);
  }

  size_t capacity, elements;
  thrust::device_ptr<TypedPool<int>> intPool;
  thrust::device_ptr<TypedPool<float>> floatPool;
  thrust::device_ptr<TypedPool<bool>> boolPool;
  thrust::device_ptr<TypedPool<thrust::tuple<bool, int>>> boolIntPool;
  thrust::device_ptr<TypedPool<vrp::models::Convolution>> convPool;
  thrust::device_ptr<TypedPool<vrp::models::JointPair>> jointPairPool;
};

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_MEMORY_DEVICEPOOL_HPP
