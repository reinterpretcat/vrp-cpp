#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace vrp::algorithms::construction {

// TODO remove base class
struct JobInsertionEvaluator {
  virtual ~JobInsertionEvaluator() = default;

protected:
  using ActivityDetail = models::solution::Activity::Detail;

  /// Specifies evaluation context.
  struct EvaluationContext final {
    /// True, if processing has to be stopped.
    bool isStopped;
    /// Violation code.
    int code = 0;
    /// Insertion index.
    size_t index = 0;
    /// Best cost.
    models::common::Cost cost = std::numeric_limits<models::common::Cost>::max();

    /// Activity detail.
    ActivityDetail detail;

    /// Creates a new context.
    static EvaluationContext empty(const models::common::Cost& cost) { return {false, 0, 0, cost, {}}; }

    /// Creates a new context from old one when insertion failed.
    static EvaluationContext fail(std::tuple<bool, int> error, const EvaluationContext& other) {
      return {std::get<0>(error), std::get<1>(error), other.index, other.cost, other.detail};
    }

    /// Creates a new context from old one when insertion worse.
    static EvaluationContext skip(const EvaluationContext& other) {
      return {other.isStopped, other.code, other.index, other.cost, other.detail};
    }

    /// Creates a new context.
    static EvaluationContext success(size_t index, const models::common::Cost& cost, const ActivityDetail& detail) {
      return {false, 0, index, cost, detail};
    }

    /// Checks whether insertion is found.
    bool isSuccess() const { return cost < std::numeric_limits<models::common::Cost>::max(); }
  };

  InsertionResult failure(const EvaluationContext& eCtx) const { return make_result_failure(eCtx.code); }

  InsertionResult success(const EvaluationContext& e,
                          const InsertionRouteContext& i,
                          const models::solution::Tour::Activity& a) const {
    return make_result_success({e.cost, a->job.value(), {{a, e.index}}, i});
  }
};
}
