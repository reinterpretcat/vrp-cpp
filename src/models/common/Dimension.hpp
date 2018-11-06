#pragma once

#include <cstdint>
#include <unordered_map>

namespace vrp::models::common {

/// Named dimension which represents:
/// * unit of measure, e.g. volume, mass, size, etc.
/// * skills
using Dimension = std::pair<std::string, std::int64_t>;

/// Multiple unit of measures.
using Dimensions = std::unordered_map<std::string, std::int64_t>;

}  // namespace vrp::models::common
