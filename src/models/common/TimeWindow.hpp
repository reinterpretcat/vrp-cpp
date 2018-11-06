#pragma once

#include "models/common/Timestamp.hpp"

namespace vrp::models::common {

struct TimeWindow final {
  Timestamp start;
  Timestamp end;
};

}
