#pragma once

#include <functional>

namespace vrp::models::problem {

struct Service;
struct Shipment;

// TODO use variant from ranges?
struct JobVisitor {
  virtual void visit(const Service&) = 0;

  virtual void visit(const Shipment&) = 0;

  virtual ~JobVisitor() = default;
};

}  // namespace vrp::models::problem
