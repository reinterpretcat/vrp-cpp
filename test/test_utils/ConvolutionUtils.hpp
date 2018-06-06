#ifndef VRP_TESTUTILS_CONVOLUTIONUTILS_HPP
#define VRP_TESTUTILS_CONVOLUTIONUTILS_HPP

#include "models/Convolution.hpp"

#include <catch/catch.hpp>

namespace vrp {
namespace test {

template<typename T>
inline thrust::device_vector<T> create(const std::initializer_list<T>& list) {
  thrust::device_vector<T> data(list.begin(), list.end());
  return std::move(data);
}

inline void compare(const vrp::models::Convolution& left, const vrp::models::Convolution& right) {
  REQUIRE(left.demand == right.demand);
  REQUIRE(left.service == right.service);

  REQUIRE(left.customers.first == right.customers.first);
  REQUIRE(left.customers.second == right.customers.second);

  REQUIRE(left.times.first == right.times.first);
  REQUIRE(left.times.second == right.times.second);

  REQUIRE(left.tasks.first == right.tasks.first);
  REQUIRE(left.tasks.second == right.tasks.second);
}

inline void compare(const vrp::models::JointPair& left, const vrp::models::JointPair& right) {
  REQUIRE(left.similarity == right.similarity);
  REQUIRE(left.completeness == right.completeness);

  compare(left.pair.first, right.pair.first);
  compare(left.pair.second, right.pair.second);
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_TESTUTILS_CONVOLUTIONUTILS_HPP