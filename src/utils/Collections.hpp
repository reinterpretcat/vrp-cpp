#include <vector>

namespace vrp::utils {

/// A helper method to add one item to a vector and pass it as rvalue in one line.
template<typename T>
std::vector<T>&&
concat(std::vector<T>& source, const T& item) {
  source.push_back(item);
  return std::move(source);
}
}