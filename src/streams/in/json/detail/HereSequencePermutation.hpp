#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "models/problem/Sequence.hpp"
#include "utils/Permutations.hpp"

#include <functional>

namespace vrp::streams::in::detail::here {

/// Generates permutations for sequence services.
/// Example:
///   Sequence with three pickups and two deliveries:
///   (p1,p2,p3) (d1,d2)
///   Possible combinations:
///   p1,p2,p3,d1,d2  p1,p2,p3,d2,d1
///   p1,p3,p2,d1,d2  p1,p3,p2,d2,d1
///
struct create_permutation_function final {
  auto operator()(int limit) {
    using namespace ranges;
    using PermutationFunc = std::function<std::vector<std::vector<int>>(const models::problem::Sequence&)>;
    // algorithms::construction::InsertionEvaluator::PermutationFunc;

    return std::make_shared<PermutationFunc>([limit](const auto& sequence) {
      // NOTE that might be a bit not perfomant..
      std::mt19937 engine(std::random_device{}());

      auto firstDeliverIndex = std::any_cast<int>(sequence.dimens.at("di"));
      auto lastPickupIndex = firstDeliverIndex - 1;
      auto size = static_cast<int>(sequence.services.size());

      return utils::generate_set_permutations{}(lastPickupIndex, size, limit, engine);
    });
  }
};
}