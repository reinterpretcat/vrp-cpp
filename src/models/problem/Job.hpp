#pragma once

#include "models/problem/Sequence.hpp"
#include "models/problem/Service.hpp"

#include <memory>
#include <range/v3/utility/variant.hpp>

namespace vrp::models::problem {

/// Represents job variant.
using Job = ranges::variant<std::shared_ptr<const Service>, std::shared_ptr<const Sequence>>;

}  // namespace vrp::models::problem
