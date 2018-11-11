#pragma once

#include "models/problem/Service.hpp"
#include "models/problem/Shipment.hpp"

#include <memory>
#include <range/v3/utility/functional.hpp>
#include <range/v3/utility/variant.hpp>

namespace vrp::models::problem {

/// Represents job variant.
using Job = ranges::variant<std::shared_ptr<const Service>, std::shared_ptr<const Shipment>>;

}  // namespace vrp::models::problem
