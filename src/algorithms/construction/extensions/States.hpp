#pragma once

#include "models/problem/Vehicle.hpp"
#include "utils/extensions/Hash.hpp"

namespace vrp::algorithms::construction {

/// Calculates vehicle key.
inline std::string
vehicleKey(const std::string& stateKey, const models::problem::Vehicle& v) {
  using namespace vrp::utils;
  using namespace vrp::models::common;
  auto hash = size_t{0} | hash_combine<Timestamp>{v.time.start} |  //
    hash_combine<Timestamp>{v.time.end} | hash_combine<Location>{v.start} |
    hash_combine<Location>{v.end.value_or(std::numeric_limits<std::uint64_t>::max())};
  return stateKey + std::to_string(hash);
}
}
