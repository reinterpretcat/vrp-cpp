#include "algorithms/heuristics/Dummy.hpp"

#include <thrust/tuple.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

__host__ __device__ TransitionCost dummy::operator()(int fromTask, int toTask, int vehicle) {
  return thrust::make_tuple(Transition(), -1);
};