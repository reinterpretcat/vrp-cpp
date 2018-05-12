#ifndef VRP_MODELS_CONVOLUTION_HPP
#define VRP_MODELS_CONVOLUTION_HPP

#include <thrust/pair.h>

namespace vrp {
namespace models {

/// Represents a group of customers served together.
struct Convolution final {
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

}
}

#endif //VRP_MODELS_CONVOLUTION_HPP
