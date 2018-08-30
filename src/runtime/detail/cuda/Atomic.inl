namespace vrp {
namespace runtime {

/// Reads old value located at the address address in global or shared memory, computes (old + val),
/// and stores the result back to memory at the same address. These three operations are performed
/// in one atomic transaction. The function returns old.
template<typename T>
__device__ inline void add(T* accumulator, T value) {
  atomicAdd(accumulator, value);
}

/// Reads old value at the address address in global or shared memory,
/// computes the maximum of old and val, and stores the result back to memory at the same address.
/// These three operations are performed in one atomic transaction.
template<typename T>
__device__ inline void max(T* oldValue, T newValue) {
  atomicMax(oldValue, newValue);
}

}  // namespace runtime
}  // namespace vrp
