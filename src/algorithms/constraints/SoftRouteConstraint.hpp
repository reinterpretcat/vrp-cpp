#pragma once

namespace vrp::algorithms::constraints {

struct SoftRouteConstraint {
  virtual double cost() const = 0;

  virtual ~SoftRouteConstraint() = default;
};

}  // namespace vrp::algorithms::constraints