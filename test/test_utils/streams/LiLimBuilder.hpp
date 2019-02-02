#pragma once

#include <istream>
#include <range/v3/all.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vrp {
namespace test {

/// Provides the way to build Li & Lim input stream programmatically.
class LiLimBuilder final {
public:
  /// Customer data.
  struct Customer {
    int id, x, y, demand, start, end, service, nope, relation;
  };

  /// Sets vehicle type.
  LiLimBuilder& setVehicle(int amount, int capacity) {
    vehicle_.first = amount;
    vehicle_.second = capacity;
    return *this;
  }

  /// Adds a new customer.
  LiLimBuilder& addCustomer(const Customer& customer) {
    customers_.push_back(customer);
    return *this;
  }

  std::stringstream build() {
    // TODO check that all needed parameters are set.
    std::stringstream ss;

    ss << vehicle_.first << " " << vehicle_.second << " 1" << std::endl;


    ranges::for_each(customers_, [&](const Customer& customer) {
      ss << customer.id << " " << customer.x << " " << customer.y << " " << customer.demand << " " << customer.start
         << " " << customer.end << " " << customer.service << " " << customer.nope << " " << customer.relation
         << std::endl;
    });

    ss << std::endl;

    return ss;
  }

private:
  std::pair<int, int> vehicle_;
  std::vector<Customer> customers_;
};

}  // namespace test
}  // namespace vrp
