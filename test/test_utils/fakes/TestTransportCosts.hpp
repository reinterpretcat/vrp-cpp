#pragma once

#include "models/costs/TransportCosts.hpp"

#include <algorithm>

namespace vrp::test {

struct TestTransportCosts final : public models::costs::TransportCosts {
  models::common::Duration duration(const models::problem::Vehicle&,
                                    const models::common::Location& from,
                                    const models::common::Location& to,
                                    const models::common::Timestamp&) const override {
    return measure<models::common::Duration>(from, to);
  }

  models::common::Distance distance(const models::problem::Vehicle&,
                                    const models::common::Location& from,
                                    const models::common::Location& to,
                                    const models::common::Timestamp&) const override {
    return measure<models::common::Distance>(from, to);
  }


private:
  template<typename Unit>
  Unit measure(const models::common::Location& from, const models::common::Location& to) const {
    return static_cast<Unit>(to > from ? to - from : from - to);
  }
};
}
