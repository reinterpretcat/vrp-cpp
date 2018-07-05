#include "algorithms/heuristics/Dummy.hpp"

#include <thrust/tuple.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

__host__ __device__ Transition dummy::operator()(const Step& step) { return Transition(); };