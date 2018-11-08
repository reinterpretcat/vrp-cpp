#pragma once

namespace vrp::algorithms::constraints {

struct HardRouteConstraint {
  virtual bool fulfilled() const = 0;

  virtual ~HardRouteConstraint() = default;
};

}  // namespace vrp::algorithms::constraints