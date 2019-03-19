#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>

namespace vrp::utils {

/// A naive implementation of date parser.
/// Returns seconds since epoch.
/// NOTE should be deprecated with C++20.
struct parse_date_from_rc3339 final {
  parse_date_from_rc3339() {
    // Calculate and store difference between local and UTC time
    time_t local = time(nullptr), utc = mktime(gmtime(&local));
    diff = utc - local;
  }

  time_t operator()(const std::string& date) const {
    std::stringstream ss(date);
    return operator()(ss);
  }

  time_t operator()(std::stringstream& date) const {
    constexpr auto Format = "%Y-%m-%dT%H:%M:%S";

    std::tm tm = {};
    date >> std::get_time(&tm, Format);

    return std::mktime(&tm) - diff;
  }

private:
  time_t diff = 0;
};
}