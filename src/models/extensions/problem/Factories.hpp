#pragma once

#include "models/problem/Service.hpp"

#include <memory>

namespace vrp::models::problem {

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

}  // namespace vrp::models::problem
