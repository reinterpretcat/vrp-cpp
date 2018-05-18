#ifndef VRP_MODELS_CONVOLUTION_HPP
#define VRP_MODELS_CONVOLUTION_HPP

#include "models/Tasks.hpp"
#include "utils/Pool.hpp"

#include <thrust/pair.h>
#include <thrust/device_vector.h>

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
  thrust::pair<int,int> customers;
  /// Time window of group.
  thrust::pair<int,int> times;
  /// Task range.
  thrust::pair<int,int> tasks;
};

/// Represent convolution collection retrieved from pool.
using Convolutions = std::unique_ptr<thrust::device_vector<vrp::models::Convolution>,
                                     vrp::utils::Pool::Deleter>;
}
}

#endif //VRP_MODELS_CONVOLUTION_HPP
