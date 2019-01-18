#pragma once

#include "models/solution/Actor.hpp"
#include "utils/extensions/Hash.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::construction {

/// Calculates actor hash.
inline size_t
actorHash(const models::solution::Actor& actor) {
  using namespace vrp::utils;
  using namespace vrp::models::common;

  const auto& detail = actor.detail;

  return size_t{0} | hash_combine<Timestamp>{detail.time.start} | hash_combine<Timestamp>{detail.time.end} |
    hash_combine<Location>{detail.start} |
    hash_combine<Location>{detail.end.value_or(std::numeric_limits<Location>::max())};
}

/// Calculates actor's unique key.
inline std::string
actorSharedKey(const std::string& key, const models::solution::Actor& actor) {
  return key + std::to_string(actorHash(actor));
}
}
