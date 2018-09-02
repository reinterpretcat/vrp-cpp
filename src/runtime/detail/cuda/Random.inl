namespace vrp {
namespace runtime {

/// Returns randomization seed
EXEC_UNIT inline unsigned int random_seed() { return static_cast<unsigned int>(clock()); }

}  // namespace runtime
}  // namespace vrp