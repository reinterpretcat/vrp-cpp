#pragma once

#include "models/problem/Service.hpp"

#include <memory>
#include <vector>

namespace vrp::models::problem {

/// Represents a job which consists of multiple sub jobs without ids.
/// All of these jobs must be performed in the order specified or none of them.
struct Sequence final {
  /// A key used to point to sequence from child service.
  constexpr static auto SeqRefDimKey = "seqRef";

  /// A list of services which must be performed in order specified.
  std::vector<std::shared_ptr<const Service>> services;

  /// Common sequence dimensions which simulates work requirements.
  vrp::models::common::Dimensions dimens;
};

}  // namespace vrp::models::problem
