#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"

namespace vrp::algorithms::construction {

inline algorithms::construction::HardActivityConstraint::Result
success() {
  return {};
}
inline algorithms::construction::HardActivityConstraint::Result
fail(int code) {
  return {{true, code}};
}
inline algorithms::construction::HardActivityConstraint::Result
stop(int code) {
  return {{false, code}};
}
}