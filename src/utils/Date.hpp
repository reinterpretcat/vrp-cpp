#pragma once

#include "models/common/Timestamp.hpp"

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

/// A naive implementation of timestamp to rfc3339 date converter.
struct timestamp_to_rc3339_string final {
  std::string operator()(models::common::Timestamp timestamp) const {
    auto l = static_cast<time_t>(timestamp);
    auto t = *std::gmtime(&l);
    std::ostringstream oss;
    oss << std::put_time(&t, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
  }
};
}