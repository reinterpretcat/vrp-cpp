namespace vrp {
namespace runtime {

/// Convert a float to a signed integer in round-to-nearest-even mode.
__device__ inline int round(float value) { return __float2int_rn(value); }

}  // namespace runtime
}  // namespace vrp
