#ifndef VRP_RUNTIME_CONFIG_HPP
#define VRP_RUNTIME_CONFIG_HPP

#ifdef RUN_ON_DEVICE
#pragma message ( "Configured to run on device." )
#include "runtime/detail/device/Config.inl"
#else
#pragma message ( "Configured to run on host." )
#include "runtime/detail/host/Config.inl"
#endif


#endif  // VRP_ITERATORS_AGGREGATES_HPP
