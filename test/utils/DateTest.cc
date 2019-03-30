#include "utils/Date.hpp"

#include <catch/catch.hpp>
#include <tuple>

using namespace vrp::utils;

namespace vrp::test {

SCENARIO("can parse date in RFC3339", "[utils][date]") {
  auto [date, expected] = GENERATE(table<std::string, int>({
    {std::string{"1970-01-01T00:00:00Z"}, 0},
    {std::string{"2018-11-19T10:00:00.000Z"}, 1542621600},
  }));

  GIVEN("date as string") {
    WHEN("parse") {
      auto result = parse_date_from_rc3339{}(date);

      THEN("returns seconds since epoch") { REQUIRE(result == expected); }
    }
  }
}

SCENARIO("can convert date from timestamp to RFC3339", "[utils][date]") {
  auto [timestamp, expected] = GENERATE(table<int, std::string>({
    {0, std::string{"1970-01-01T00:00:00Z"}},
    {1542621600, std::string{"2018-11-19T10:00:00Z"}},
  }));

  GIVEN("date as timestamp") {
    WHEN("convert") {
      auto result = timestamp_to_rc3339_string{}(timestamp);

      THEN("returns seconds since epoch") { REQUIRE(result == expected); }
    }
  }
}
}