#pragma once

#include "models/solution/Activity.hpp"

#include <any>

namespace vrp::models::solution {

/// Retrieves original job from activity.
struct retrieve_job final {
  std::optional<problem::Job> operator()(const Activity& activity) const {
    if (!activity.service.has_value()) return {};

    auto seqRef = activity.service.value()->dimens.find(problem::Sequence::SeqRefDimKey);

    return seqRef != activity.service.value()->dimens.end() ? std::any_cast<problem::Job>(seqRef->second)
                                                            : as_job(activity.service.value());
  }
};
}