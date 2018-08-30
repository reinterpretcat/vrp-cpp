#include <mutex>

namespace vrp {
namespace runtime {

/// TODO use C++11 atomics?
static std::mutex mx;

template<typename T>
__host__ inline void add(T* accumulator, T value) {
  std::lock_guard<std::mutex> lock{mx};
  *accumulator += value;
}

template<typename T>
__host__ inline void max(T* oldValue, T newValue) {
  std::lock_guard<std::mutex> lock{mx};
  if (*oldValue < newValue) *oldValue = newValue;
}


}  // namespace runtime
}  // namespace vrp
