#pragma once


#include "models/problem/Job.hpp"

namespace vrp::models::problem {

struct job_distance final {
  double operator()(const Job& lhs, const Job& rhs) const {
    // TODO
    return 0.0;
  }
};
}