#ifndef VRP_UTILS_POOL_HPP
#define VRP_UTILS_POOL_HPP

#include <functional>
#include <typeindex>
#include <typeinfo>
#include <memory>
#include <stack>
#include <vector>
#include <unordered_map>

namespace vrp {
namespace utils {

/// Implements object pool for arbitrary types.
class Pool final {
 public:
  /// Specifies signature of unique_ptr deleter function.
  using Deleter = std::function<void(void *)>;

  /// Implements object pool for specific type.
  template<class T>
  class TypedPool {
   public:
    using ptr_type = std::unique_ptr<T, Deleter>;

    TypedPool() = default;
    virtual ~TypedPool() = default;

    TypedPool(const TypedPool &) = delete;
    TypedPool &operator=(const TypedPool &) = delete;

    ptr_type acquire() {
      auto obj = pool_.empty() ? new T() : pool_.top().release();
      ptr_type tmp(obj, [this](void *ptr) {
        this->release(std::unique_ptr<T>(static_cast<T*>(ptr)));
      });

      if (!pool_.empty())
        pool_.pop();

      return std::move(tmp);
    }

    void release(std::unique_ptr<T> t) {
      pool_.push(std::move(t));
    }

   private:
    std::stack<std::unique_ptr<T>> pool_;
  };

  /// Returns object from pool.
  template<typename T>
  typename TypedPool<T>::ptr_type acquire() {
    return pool<T>()->acquire();
  }

  /// Returns container from pool.
  template<typename Container>
  typename TypedPool<Container>::ptr_type acquire(size_t size) {
    auto container = pool<Container>()->acquire();
    container->resize(size);
    return std::move(container);
  }

 private:

  /// Returns object from pool.
  template<typename T>
  std::shared_ptr<TypedPool<T>> pool() {
    auto it = poolMap_.insert(std::make_pair(std::type_index(typeid(T)),
                                             std::shared_ptr<void>(new TypedPool<T>())));
    return std::static_pointer_cast<TypedPool<T>>(it.first->second);
  }

  std::unordered_map<std::type_index, std::shared_ptr<void>> poolMap_;
};

}
}

#endif //VRP_UTILS_POOL_HPP
