#pragma once

#include "models/costs/TransportCosts.hpp"

#include <algorithm>

namespace vrp::test {

struct TestTransportCosts final : public models::costs::TransportCosts {
  models::common::Duration duration(const models::problem::Actor& actor,
                                    const models::common::Location& from,
                                    const models::common::Location& to,
                                    const models::common::Timestamp& departure) const override {
    return static_cast<models::common::Duration>(std::abs(static_cast<long>(to - from)));
  }

  models::common::Cost cost(const models::problem::Actor& actor,
                            const models::common::Location& from,
                            const models::common::Location& to,
                            const models::common::Timestamp& departure) const override {
    return static_cast<models::common::Cost>(std::abs(static_cast<long>(to - from)));
  }
};
}
