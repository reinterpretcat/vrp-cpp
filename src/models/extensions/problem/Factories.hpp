#pragma once

#include "models/problem/Actor.hpp"
#include "models/problem/Driver.hpp"
#include "models/problem/Job.hpp"
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
  build_service& withId(std::string&& id) {
    service_.id = id;
    return *this;
  }

  build_service& withLocation(std::optional<common::Location>&& location) {
    service_.location = location;
    return *this;
  }

  build_service& withTimes(std::vector<common::TimeWindow>&& times) {
    service_.times = times;
    return *this;
  }

  build_service& withDimensions(common::Dimensions&& dimens) {
    service_.dimens = dimens;
    return *this;
  }

  Service&& owned() { return std::move(service_); }

  std::shared_ptr<Service> shared() { return std::make_shared<Service>(std::move(service_)); }

private:
  Service service_;
};

/// A helper class to build vehicle;
class build_vehicle {
public:
  build_vehicle& withId(std::string&& id) {
    vehicle_.id = id;
    return *this;
  }

  build_vehicle& withProfile(std::string&& profile) {
    vehicle_.profile = profile;
    return *this;
  }

  build_vehicle& withCosts(Costs&& costs) {
    vehicle_.costs = costs;
    return *this;
  }

  build_vehicle& withSchedule(common::Schedule&& schedule) {
    vehicle_.schedule = schedule;
    return *this;
  }

  build_vehicle& withDimensions(common::Dimensions&& dimensions) {
    vehicle_.dimensions = dimensions;
    return *this;
  }

  build_vehicle& withStart(common::Location&& start) {
    vehicle_.start = start;
    return *this;
  }

  build_vehicle& withEnd(std::optional<common::Location>&& end) {
    vehicle_.end = end;
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
  build_driver& withCosts(Costs&& costs) {
    driver_.costs = costs;
    return *this;
  }

  build_driver& withSchedule(common::Schedule&& schedule) {
    driver_.schedule = schedule;
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
  build_actor& withDriver(const std::shared_ptr<const Driver>& driver) {
    actor_.driver = driver;
    return *this;
  }

  build_actor& withVehicle(const std::shared_ptr<const Vehicle>& vehicle) {
    actor_.vehicle = vehicle;
    return *this;
  }

  Actor&& owned() { return std::move(actor_); }

  std::shared_ptr<Actor> shared() { return std::make_shared<Actor>(std::move(actor_)); }

private:
  Actor actor_;
};

}  // namespace vrp::models::problem
