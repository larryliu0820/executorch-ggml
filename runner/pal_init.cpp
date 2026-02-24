// Ensure et_pal_init() runs before main().
// The ExecuTorch posix PAL aborts in debug builds if any PAL function
// (e.g. et_pal_current_ticks) is called before et_pal_init().
#include <executorch/runtime/platform/platform.h>

namespace {
struct PalInitializer {
  PalInitializer() { et_pal_init(); }
};
static PalInitializer pal_init_;
} // namespace
