#pragma once

#include <any>
#include <cstdint>
#include <unordered_map>

namespace vrp::models::common {

/// Named dimension which can represents anything:
/// * unit of measure, e.g. volume, mass, size, etc.
/// * set of skills
/// * tag
using Dimension = std::pair<std::string, std::any>;

/// Multiple named dimensions.
using Dimensions = std::unordered_map<std::string, std::any>;

}  // namespace vrp::models::common
