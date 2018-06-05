#ifndef VRP_MODELS_CONVOLUTION_HPP
#define VRP_MODELS_CONVOLUTION_HPP

#include "models/Tasks.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace models {

/// Represents a group of customers served together.
struct Convolution final {
  /// Base index
  int base;
  /// Total demand of all customers in group.
  int demand;
  /// Total service time needed to serve all customers in group.
  int service;
  /// First and last customer in group.
  thrust::pair<int, int> customers;
  /// Time window of group.
  thrust::pair<int, int> times;
  /// Task range.
  thrust::pair<int, int> tasks;
};

/// Represents a convolution joint pair.
struct JointPair final {
  /// Amount of shared customers.
  int similarity;
  /// Amount of unique customers served by convolution pair.
  int completeness;
  /// A pair constructed from two different convolutions.
  thrust::pair<vrp::models::Convolution, vrp::models::Convolution> pair;
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_CONVOLUTION_HPP
