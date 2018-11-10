#pragma once

#include "models/problem/Job.hpp"
#include "models/problem/Service.hpp"
#include "models/problem/Shipment.hpp"

namespace vrp::models::problem {

/// Simplifies the way to visit jobs.
/// NOTE it would be nice to use std::variant and std::visit, but there is
/// some problems with it within clang. So, use inheritance and classic visitor
/// pattern implementation.
template<typename T>
inline T
visit_job(std::function<T(const Service&)> serviceFunctor,
          std::function<T(const Shipment&)> shipmentFunctor,
          const Job& job) {
  using ServiceFunctor = std::function<T(const Service&)>;
  using ShipmentFunctor = std::function<T(const Shipment&)>;

  struct StatefulJobVisitor : private JobVisitor {
    T state;

    StatefulJobVisitor(ServiceFunctor serviceFunctor, ShipmentFunctor shipmentFunctor) :
      state(), serviceFunctor_(serviceFunctor), shipmentFunctor_(shipmentFunctor) {}

    void visit(const Service& service) override { state = serviceFunctor_(service); }

    void visit(const Shipment& shipment) override { state = shipmentFunctor_(shipment); }

  private:
    ServiceFunctor serviceFunctor_;
    ShipmentFunctor shipmentFunctor_;
  };

  auto visitor = StatefulJobVisitor{serviceFunctor, shipmentFunctor};
  job.accept(visitor);
  return visitor.state;
}
}