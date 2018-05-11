#ifndef VRP_UTILS_POOL_HPP
#define VRP_UTILS_POOL_HPP

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

  /// Implements object pool for specific type.
  template <class T>
  class TypedPool {
   private:
    struct External_Deleter {
      explicit External_Deleter(std::weak_ptr<TypedPool<T>* > pool)
          : pool_(pool) {}

      void operator()(T* ptr) {
        if (auto pool_ptr = pool_.lock()) {
          try {
            (*pool_ptr.get())->add(std::unique_ptr<T>{ptr});
            return;
          } catch(...) {}
        }
        std::default_delete<T>{}(ptr);
      }
     private:
      std::weak_ptr<TypedPool<T>* > pool_;
    };
   public:
    using ptr_type = std::unique_ptr<T, External_Deleter >;

    TypedPool() : this_ptr_(new TypedPool<T>*(this)) {}
    virtual ~TypedPool() = default;

    TypedPool(const TypedPool&) = delete;
    TypedPool&operator=(const TypedPool&) = delete;

    void add(std::unique_ptr<T> t) {
      pool_.push(std::move(t));
    }

    ptr_type acquire() {
      ptr_type tmp(pool_.empty() ? new T() : pool_.top().release(),
                   External_Deleter{std::weak_ptr<TypedPool<T>*>{ this_ptr_ }});

      if (!pool_.empty())
        pool_.pop();

      return std::move(tmp);
    }

   private:
    std::shared_ptr<TypedPool<T>*> this_ptr_;
    std::stack<std::unique_ptr<T>> pool_;
  };

  /// Returns object from pool.
  template <typename T>
  typename TypedPool<T>::ptr_type acquire() {
    return pool<T>()->acquire();
  }

  /// Returns container from pool.
  template <typename Container>
  typename TypedPool<Container>::ptr_type acquire(size_t size) {
    auto container = pool<Container>()->acquire();
    container->resize(size);
    return std::move(container);
  }

 private:

  /// Returns object from pool.
  template <typename T>
  std::shared_ptr<TypedPool<T>> pool() {
    auto it = poolMap_.insert(std::make_pair(std::type_index(typeid(T)),
                                             std::shared_ptr<void>(new TypedPool<T>())));
    return std::static_pointer_cast<TypedPool<T>>(it.first->second);
  }

  std::unordered_map<std::type_index,std::shared_ptr<void>> poolMap_;
};

}
}

#endif //VRP_UTILS_POOL_HPP
