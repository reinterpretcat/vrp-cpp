#include "algorithms/heuristics/Dummy.hpp"

#include <thrust/tuple.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

ANY_EXEC_UNIT Transition dummy::operator()(const Step& step) { return Transition(); };