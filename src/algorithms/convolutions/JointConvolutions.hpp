#ifndef VRP_ALGORITHMS_CONVOLUTIONS_JOINTCONVOLUTIONS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_JOINTCONVOLUTIONS_HPP

#include "algorithms/convolutions/Models.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Provides the way to create joint pairs of convolutions with
/// additional characteristics.
struct create_joint_convolutions final {

  JointPairs operator()(const vrp::models::Problem &problem,
                        vrp::models::Tasks &tasks,
                        const Settings &settings,
                        const vrp::models::Convolutions &left,
                        const vrp::models::Convolutions &right) const;
};

}
}
}

#endif //VRP_ALGORITHMS_CONVOLUTIONS_JOINTCONVOLUTIONS_HPP
