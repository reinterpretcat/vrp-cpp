#ifndef VRP_UTILS_SOLOMONBUILDER_HPP
#define VRP_UTILS_SOLOMONBUILDER_HPP

#include <string>
#include <istream>
#include <utility>
#include <vector>
#include <sstream>
#include <algorithm>

namespace vrp {
namespace test {

/// Provides the way to build solomon input stream programmatically.
class SolomonBuilder final {
 public:
  /// Customer data.
  struct Customer {
    int id, x, y, demand, start, end, service;
  };

  /// Sets problem title.
  SolomonBuilder &setTitle(const std::string &title) {
    title_ = title;
    return *this;
  }

  /// Sets vehicle type.
  SolomonBuilder &setVehicle(int amount, int capacity) {
    vehicle_.first = amount;
    vehicle_.second = capacity;
    return *this;
  }

  /// Adds a new customer.
  SolomonBuilder &addCustomer(const Customer &customer) {
    customers_.push_back(customer);
    return *this;
  }

  std::stringstream build() {
    // TODO check that all needed parameters are set.
    std::stringstream ss;

    ss << title_ << std::endl << std::endl;

    ss << "VEHICLE" << std::endl << "NUMBER     CAPACITY" << std::endl;
    ss << "  " << vehicle_.first <<"          " << vehicle_.second << std::endl
       << std::endl;

    ss << "CUSTOMER" << std::endl
       << "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME   DUE DATE   SERVICE TIME" << std::endl
       << std::endl;

    std::for_each(customers_.begin(), customers_.end(), [&](const Customer& customer) {
      ss << customer.id << " "  << customer.x << " " << customer.y << " " << customer.demand << " "
         << customer.start << " " << customer.end << " " << customer.service << " " << std::endl;
    });

    ss << std::endl;

    return ss;
  }

 private:
  std::string title_;
  std::pair<int,int> vehicle_;
  std::vector<Customer> customers_;
};

}
}

#endif //VRP_UTILS_SOLOMONBUILDER_HPP
