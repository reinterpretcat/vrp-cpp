#pragma once

#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Driver.hpp"
#include "models/problem/Job.hpp"
#include "models/problem/Sequence.hpp"
#include "models/problem/Service.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>
#include <range/v3/utility/variant.hpp>

namespace vrp::models::problem {

/// A helper class to create service job.
class build_service {
public:
  build_service& details(std::vector<problem::Service::Detail>&& value) {
    service_.details = std::move(value);
    return *this;
  }

  build_service& dimens(common::Dimensions&& value) {
    service_.dimens = std::move(value);
    return *this;
  }

  Service&& owned() { return std::move(service_); }

  std::shared_ptr<Service> shared() { return std::make_shared<Service>(std::move(service_)); }

protected:
  Service service_;
};

/// A helper class to create sequence job.
class build_sequence {
public:
  build_sequence& dimens(common::Dimensions&& value) {
    sequence_->dimens = std::move(value);
    return *this;
  }

  build_sequence& service(const std::shared_ptr<Service>& value) {
    value->dimens[Sequence::SeqRefDimKey] = as_job(sequence_);
    sequence_->services.push_back(value);
    return *this;
  }

  build_sequence& services(std::vector<std::shared_ptr<Service>> services) {
    ranges::for_each(services, [&](const auto& s) { service(s); });
    return *this;
  }

  std::shared_ptr<Sequence> shared() { return sequence_; }

protected:
  std::shared_ptr<Sequence> sequence_ = std::make_shared<Sequence>();
};

/// A helper class to build vehicle;
class build_vehicle {
public:
  build_vehicle& profile(common::Profile value) {
    vehicle_.profile = value;
    return *this;
  }

  build_vehicle& costs(const Costs& value) {
    vehicle_.costs = value;
    return *this;
  }

  build_vehicle& details(std::vector<Vehicle::Detail>&& value) {
    vehicle_.details = std::move(value);
    return *this;
  }

  build_vehicle& dimens(common::Dimensions&& value) {
    vehicle_.dimens = std::move(value);
    return *this;
  }

  Vehicle&& owned() { return std::move(vehicle_); }

  std::shared_ptr<Vehicle> shared() { return std::make_shared<Vehicle>(std::move(vehicle_)); }

protected:
  Vehicle vehicle_;
};

/// A helper class to build driver.
class build_driver {
public:
  build_driver& dimens(common::Dimensions&& value) {
    driver_.dimens = std::move(value);
    return *this;
  }

  build_driver& costs(const Costs& value) {
    driver_.costs = value;
    return *this;
  }

  Driver&& owned() { return std::move(driver_); }

  std::shared_ptr<Driver> shared() { return std::make_shared<Driver>(std::move(driver_)); }

protected:
  Driver driver_;
};

}  // namespace vrp::models::problem
