#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>

namespace nlohmann {
template<typename T>
struct adl_serializer<std::optional<T>> {
  static void to_json(json& j, const std::optional<T>& opt) {
    if (opt)
      j = *opt;
    else
      j = nullptr;
  }

  static void from_json(const json& j, std::optional<T>& opt) {
    if (j.is_null())
      opt = {};
    else
      opt = j.get<T>();
  }
};
}

namespace vrp::streams::in {

namespace detail {

template<typename T>
void
readOptional(const nlohmann::json& j, const std::string& key, T& v) {
  if (j.find(key) != j.end()) j.at(key).get_to(v);
}

// region Common

using Location = std::uint64_t;
using Timestamp = double;

struct TimeWindow {
  Timestamp start;
  Timestamp end;
};

void
from_json(const nlohmann::json& j, TimeWindow& tw) {
  j.at("start").get_to(tw.start);
  j.at("end").get_to(tw.end);
}

// endregion
}
}