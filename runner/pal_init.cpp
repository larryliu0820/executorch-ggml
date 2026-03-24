// Override ExecuTorch PAL log function to print to stderr.
// The default PAL in libexecutorch.so is a no-op for logging.
// We override via strong symbol (weak in the shared lib).
#include <cstdio>
#include <cstdlib>
#include <executorch/runtime/platform/platform.h>

// Provide a strong definition that overrides the weak one in libexecutorch.
extern "C" {
void et_pal_emit_log_message(
    et_timestamp_t /*timestamp*/,
    et_pal_log_level_t level,
    const char* /*filename*/,
    const char* /*function*/,
    size_t /*line*/,
    const char* message,
    size_t /*length*/) {
  const char* tag = "?";
  switch (level) {
    case kDebug: tag = "D"; break;
    case kInfo:  tag = "I"; break;
    case kError: tag = "E"; break;
    case kFatal: tag = "F"; break;
    default: break;
  }
  fprintf(stderr, "[ET/%s] %s\n", tag, message);
}
}
