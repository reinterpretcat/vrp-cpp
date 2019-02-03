# Description

A prototype of VRP solver


https://github.com/justusc/FindTBB

# Install

        # build tbb from https://github.com/philipp-classen/tbb-static-linking-tutorial
        make extra_inc=big_iron.inc


# TODO

 * handle jobs which cannot be assigned
 * simplify registry
 * check movement syntax within builders to avoid copying

 * remove string selects job from activity (does not work for sequence)
 * remove type: use optional as start-end marker?


 * improve jobs distances logic?
