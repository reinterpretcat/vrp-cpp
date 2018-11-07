#pragma once

#include <any>
#include <cstdint>
#include <unordered_map>

namespace vrp::models::common {

/// Dimension which can represents anything:
/// * unit of measure, e.g. volume, mass, size, etc.
/// * set of skills
/// * tag
using Dimension = std::any;

/// Multiple named dimensions.
using Dimensions = std::unordered_map<std::string, Dimension>;

}  // namespace vrp::models::common
