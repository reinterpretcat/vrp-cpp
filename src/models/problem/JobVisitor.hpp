#pragma once

namespace vrp::models::problem {

struct Service;
struct Shipment;

struct JobVisitor {

  virtual void visit(const Service&) = 0;

  virtual void visit(const Shipment&) = 0;

  virtual ~JobVisitor() = default;
};

}
