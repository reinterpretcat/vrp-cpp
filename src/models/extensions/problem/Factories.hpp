#pragma once

#include "models/problem/Actor.hpp"
#include "models/problem/Driver.hpp"
#include "models/problem/Job.hpp"
#include "models/problem/JobDetail.hpp"
#include "models/problem/Service.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>
#include <range/v3/utility/variant.hpp>

namespace vrp::models::problem {

/// Creates job from service.
inline Job
as_job(const std::shared_ptr<const Service>& service) {
  return {ranges::emplaced_index<0>, service};
}

/// Creates job from shipment.
inline Job
as_job(const std::shared_ptr<const Shipment>& shipment) {
  return {ranges::emplaced_index<1>, shipment};
}

/// A helper class to create service job.
class build_service {
public:
  build_service& id(std::string&& value) {
    service_.id = value;
    return *this;
  }

  build_service& details(std::vector<problem::JobDetail>&& value) {
    service_.details = value;
    return *this;
  }

  build_service& dimensions(common::Dimensions&& value) {
    service_.dimens = value;
    return *this;
  }

  Service&& owned() { return std::move(service_); }

  std::shared_ptr<Service> shared() { return std::make_shared<Service>(std::move(service_)); }

private:
  Service service_;
};

class build_detail {
public:
  build_detail& location(common::Location value) {
    detail_.location = value;
    return *this;
  }

  build_detail& duration(common::Duration value) {
    detail_.duration = value;
    return *this;
  }

  build_detail& times(std::vector<common::TimeWindow>&& value) {
    detail_.times = value;
    return *this;
  }

  JobDetail&& owned() { return std::move(detail_); }

  std::shared_ptr<JobDetail> shared() { return std::make_shared<JobDetail>(std::move(detail_)); }

private:
  JobDetail detail_;
};

/// A helper class to build vehicle;
class build_vehicle {
public:
  build_vehicle& id(std::string&& value) {
    vehicle_.id = value;
    return *this;
  }

  build_vehicle& profile(std::string&& value) {
    vehicle_.profile = value;
    return *this;
  }

  build_vehicle& costs(const Costs& value) {
    vehicle_.costs = value;
    return *this;
  }

  build_vehicle& time(common::TimeWindow&& value) {
    vehicle_.time = value;
    return *this;
  }

  build_vehicle& dimensions(common::Dimensions&& value) {
    vehicle_.dimensions = value;
    return *this;
  }

  build_vehicle& start(common::Location value) {
    vehicle_.start = value;
    return *this;
  }

  build_vehicle& end(common::Location value) {
    vehicle_.end = value;
    return *this;
  }

  Vehicle&& owned() { return std::move(vehicle_); }

  std::shared_ptr<Vehicle> shared() { return std::make_shared<Vehicle>(std::move(vehicle_)); }

private:
  Vehicle vehicle_;
};

/// A helper class to build driver.
class build_driver {
public:
  build_driver& costs(const Costs& value) {
    driver_.costs = value;
    return *this;
  }

  Driver&& owned() { return std::move(driver_); }

  std::shared_ptr<Driver> shared() { return std::make_shared<Driver>(std::move(driver_)); }

private:
  Driver driver_;
};

/// A helper class to build actor.
class build_actor {
public:
  build_actor& driver(const std::shared_ptr<const Driver>& value) {
    actor_.driver = value;
    return *this;
  }

  build_actor& vehicle(const std::shared_ptr<const Vehicle>& value) {
    actor_.vehicle = value;
    return *this;
  }

  Actor&& owned() { return std::move(actor_); }

  std::shared_ptr<Actor> shared() { return std::make_shared<Actor>(std::move(actor_)); }

private:
  Actor actor_;
};

}  // namespace vrp::models::problem
